//! memchat — a TUI for chatting with a memory-grounded agent.
//!
//! You type messages; each one is:
//!   1. ingested through the LLM-free gate (novelty + salience + correction),
//!   2. used as a recall query against the store (BM25 + dense + RRF + rerank),
//!   3. answered by an LLM (gpt-5-mini) grounded in the recalled memories, or
//!      echoed from memory when no OPENAI_API_KEY is set,
//!   4. the assistant reply is itself ingested so the conversation has
//!      continuity — the agent remembers what it just told you.
//!
//! Layout: chat transcript (left) + recalled-memory panel (right) + status bar
//! + input box. Scroll the transcript with Up/Down/PgUp/PgDown. Toggle help
//! with ?. Commands start with `:` (see :help).
//!
//! Run:  cargo run --bin memchat -- [--agent <id>] [--fresh]
//!        OPENAI_API_KEY=sk-...  required for grounded LLM replies (else echo).

use anyhow::{Context, Result};
use clap::Parser;
use consolidation::ConsolidationMode;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap},
    Terminal,
};
use serde_json::{json, Value};
use std::io::{self, Stdout};
use std::sync::Arc;
use std::time::Duration;
use store::{Config, Decision, Hit, Store};

// ─── config / state ──────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "memchat", about = "Memory-grounded chat TUI", version)]
struct Cli {
    #[arg(long, default_value = "memoryd.toml")]
    config: String,
    #[arg(long, default_value = "dev")]
    agent: String,
    /// Drop + recreate the messages table at 384-dim for a clean slate.
    #[arg(long, default_value_t = false)]
    fresh: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role { User, Assistant }

#[derive(Clone)]
struct Turn {
    role: Role,
    text: String,
    /// One-line meta under the bubble: gate decision / recall count / model.
    meta: String,
    /// Recalled memories shown for this turn (assistant turns only).
    memories: Vec<Hit>,
    /// Pending assistant turns render a spinner instead of text.
    pending: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase { Idle, Recalling, Thinking }

struct App {
    agent: String,
    scope: String,
    mode: ConsolidationMode,
    model: String, // consolidation model name
    k: usize,
    turns: Vec<Turn>,
    input: String,
    cursor: usize, // byte offset
    scroll: usize, // lines hidden from the bottom (0 = pinned to latest)
    phase: Phase,
    pending: usize,
    spinner: usize,
    show_help: bool,
    log: String, // last status/log line
    llm_on: bool,
}

impl App {
    fn new(agent: String, llm_on: bool) -> Self {
        Self {
            agent,
            scope: "private".into(),
            mode: ConsolidationMode::None,
            model: "extractive".into(),
            k: 5,
            turns: Vec::new(),
            input: String::new(),
            cursor: 0,
            scroll: 0,
            phase: Phase::Idle,
            pending: 0,
            spinner: 0,
            show_help: false,
            log: String::new(),
            llm_on,
        }
    }

    fn push_log(&mut self, s: impl Into<String>) {
        self.log = s.into();
    }
}

// ─── UI events (from terminal + background tasks) ────────────────────────────

enum UiEvent {
    Term(Event),
    UserIngested { turn: usize, decision: Decision },
    RecallDone { turn: usize, hits: Vec<Hit> },
    AnswerDone { turn: usize, text: String, reply_id: Option<i64> },
    ReplyIngestError { why: String },
    Error(String),
}

// ─── main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load(&cli.config).map_err(|e| {
        anyhow::anyhow!("failed to load config '{}': {e}\nSee README for the memoryd.toml template.", cli.config)
    })?;
    let store = Arc::new(Store::connect(cfg).await?);
    if cli.fresh {
        store.reset_for_dim(384).await?;
    }

    let openai_key = std::env::var("OPENAI_API_KEY").ok().filter(|s| !s.is_empty());
    let llm_on = openai_key.is_some();

    let mut app = App::new(cli.agent, llm_on);
    app.push_log(if llm_on {
        "LLM on (gpt-5-mini). Type a message, or :help for commands.".to_string()
    } else {
        "OPENAI_API_KEY not set — echo mode (answers list recalled memories). Set the key for grounded replies.".to_string()
    });

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<UiEvent>();
    spawn_input_thread(tx.clone());

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let _guard = TerminalGuard;

    run_app(&mut terminal, &mut app, store, openai_key, tx, &mut rx).await
}

/// RAII guard so the terminal is restored even on panic / early return.
struct TerminalGuard;
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }
}

fn spawn_input_thread(tx: tokio::sync::mpsc::UnboundedSender<UiEvent>) {
    std::thread::spawn(move || {
        // poll-based: short timeout so the thread is responsive to process exit.
        loop {
            match event::poll(Duration::from_millis(100)) {
                Ok(true) => {
                    if let Ok(ev) = event::read() {
                        if tx.send(UiEvent::Term(ev)).is_err() {
                            break;
                        }
                    }
                }
                Ok(false) => {}
                Err(e) => {
                    let _ = tx.send(UiEvent::Error(format!("input poll: {e}")));
                    std::thread::sleep(Duration::from_millis(500));
                }
            }
        }
    });
}

// ─── event loop ──────────────────────────────────────────────────────────────

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut App,
    store: Arc<Store>,
    openai_key: Option<String>,
    tx: tokio::sync::mpsc::UnboundedSender<UiEvent>,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<UiEvent>,
) -> Result<()> {
    let mut tick = tokio::time::interval(Duration::from_millis(120));
    loop {
        terminal.draw(|f| render(f, app))?;

        tokio::select! {
            ev = rx.recv() => {
                let Some(ev) = ev else { break };
                if handle_event(ev, app, &store, &openai_key, &tx).await? {
                    break;
                }
            }
            _ = tick.tick() => {
                app.spinner = (app.spinner + 1) % 4;
            }
        }
    }
    Ok(())
}

/// Returns `true` to quit.
async fn handle_event(
    ev: UiEvent,
    app: &mut App,
    store: &Arc<Store>,
    openai_key: &Option<String>,
    tx: &tokio::sync::mpsc::UnboundedSender<UiEvent>,
) -> Result<bool> {
    match ev {
        UiEvent::Term(Event::Key(k)) => handle_key(k, app, store, openai_key, tx).await,
        UiEvent::Term(Event::Resize(_, _)) => Ok(false),
        UiEvent::Term(_) => Ok(false),

        UiEvent::UserIngested { turn, decision } => {
            if let Some(t) = app.turns.get_mut(turn) {
                if decision.admitted {
                    t.meta = format!("stored id={} · novelty {:.2} · salience {:.2} · {}",
                        decision.id.unwrap(), decision.novelty, decision.salience, decision.reason);
                } else {
                    t.meta = format!("dropped · {} · novelty {:.2} · salience {:.2}",
                        decision.reason, decision.novelty, decision.salience);
                }
            }
            Ok(false)
        }
        UiEvent::RecallDone { turn, hits } => {
            app.phase = Phase::Thinking;
            if let Some(t) = app.turns.get_mut(turn) {
                t.memories = hits.clone();
                t.meta = format!("recalled {} · thinking…", hits.len());
            }
            app.scroll = 0;
            Ok(false)
        }
        UiEvent::AnswerDone { turn, text, reply_id } => {
            if let Some(t) = app.turns.get_mut(turn) {
                t.text = text;
                t.pending = false;
                let idstr = reply_id.map(|i| format!(" · stored id={i}")).unwrap_or_default();
                let mem = t.memories.len();
                t.meta = format!("recalled {mem}{idstr}");
            }
            app.pending = app.pending.saturating_sub(1);
            app.phase = if app.pending == 0 { Phase::Idle } else { Phase::Thinking };
            app.scroll = 0;
            Ok(false)
        }
        UiEvent::ReplyIngestError { why } => {
            app.push_log(format!("reply ingest failed: {why}"));
            Ok(false)
        }
        UiEvent::Error(why) => {
            app.push_log(format!("error: {why}"));
            Ok(false)
        }
    }
}

// ─── input handling ──────────────────────────────────────────────────────────

async fn handle_key(
    k: KeyEvent,
    app: &mut App,
    store: &Arc<Store>,
    openai_key: &Option<String>,
    tx: &tokio::sync::mpsc::UnboundedSender<UiEvent>,
) -> Result<bool> {
    if app.show_help {
        if matches!(k.code, KeyCode::Esc | KeyCode::Enter | KeyCode::Char('?') | KeyCode::Char('q')) {
            app.show_help = false;
        }
        return Ok(false);
    }

    match k.code {
        KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => return Ok(true),
        KeyCode::Char('d') if k.modifiers.contains(KeyModifiers::CONTROL) => return Ok(true),
        KeyCode::Esc => return Ok(true),

        KeyCode::Char('?') if app.input.is_empty() => {
            app.show_help = true;
            return Ok(false);
        }

        // scrolling
        KeyCode::Up => { app.scroll = app.scroll.saturating_add(1); }
        KeyCode::Down => { app.scroll = app.scroll.saturating_sub(1); }
        KeyCode::PageUp => { app.scroll = app.scroll.saturating_add(10); }
        KeyCode::PageDown => { app.scroll = app.scroll.saturating_sub(10); }

        // line editing
        KeyCode::Left => {
            if let Some(i) = prev_char_boundary(&app.input, app.cursor) {
                app.cursor = i;
            }
        }
        KeyCode::Right => {
            if let Some(i) = next_char_boundary(&app.input, app.cursor) {
                app.cursor = i;
            }
        }
        KeyCode::Home if !app.input.is_empty() => app.cursor = 0,
        KeyCode::End if !app.input.is_empty() => app.cursor = app.input.len(),
        KeyCode::Home => { app.scroll = usize::MAX; }
        KeyCode::End => { app.scroll = 0; }

        KeyCode::Backspace => {
            if let Some(i) = prev_char_boundary(&app.input, app.cursor) {
                app.input.replace_range(i..app.cursor, "");
                app.cursor = i;
            }
        }
        KeyCode::Delete => {
            if let Some(i) = next_char_boundary(&app.input, app.cursor) {
                app.input.replace_range(app.cursor..i, "");
            }
        }

        KeyCode::Enter => {
            let line = app.input.trim().to_string();
            app.input.clear();
            app.cursor = 0;
            if line.is_empty() {
                return Ok(false);
            }
            if line.starts_with(':') {
                handle_command(&line, app, store, tx).await?;
            } else {
                send_chat(&line, app, store, openai_key, tx);
            }
        }

        KeyCode::Char(ch) => {
            app.input.insert(app.cursor, ch);
            app.cursor += ch.len_utf8();
        }
        _ => {}
    }
    Ok(false)
}

async fn handle_command(
    line: &str,
    app: &mut App,
    store: &Arc<Store>,
    tx: &tokio::sync::mpsc::UnboundedSender<UiEvent>,
) -> Result<()> {
    let mut parts = line[1..].split_whitespace();
    let cmd = parts.next().unwrap_or("");
    let a0 = parts.next();
    let a1 = parts.next();
    match cmd {
        "help" | "h" => app.show_help = true,
        "quit" | "q" | "exit" => {
            // signal quit by sending a synthetic key event is awkward; just log
            app.push_log("press Ctrl-C or Esc to quit");
        }
        "agent" => {
            if let Some(a) = a0 {
                app.agent = a.to_string();
                app.push_log(format!("agent → {a}"));
            } else {
                app.push_log("usage: :agent <id>");
            }
        }
        "scope" => {
            match a0 {
                Some("private") | Some("shared") => {
                    app.scope = a0.unwrap().to_string();
                    app.push_log(format!("scope → {}", app.scope));
                }
                _ => app.push_log("usage: :scope private|shared"),
            }
        }
        "mode" => {
            if let Some(m) = a0 {
                match m.parse::<ConsolidationMode>() {
                    Ok(m) => { app.mode = m; app.push_log(format!("mode → {m:?}").to_lowercase()); }
                    Err(_) => app.push_log("usage: :mode none|lazy|eager"),
                }
            } else {
                app.push_log("usage: :mode none|lazy|eager");
            }
        }
        "model" => {
            if let Some(m) = a0 {
                app.model = m.to_string();
                app.push_log(format!("consolidation model → {m}"));
            } else {
                app.push_log("usage: :model extractive|openai");
            }
        }
        "k" => {
            if let Some(n) = a0.and_then(|s| s.parse::<usize>().ok()) {
                app.k = n;
                app.push_log(format!("k → {n}"));
            } else {
                app.push_log("usage: :k <n>");
            }
        }
        "seed" => do_seed(app, store).await?,
        "consolidate" | "cons" => {
            if let Some(m) = a0 { app.model = m.to_string(); }
            spawn_consolidate(store.clone(), app.model.clone(), tx.clone());
            app.push_log(format!("consolidating (model={})…", app.model));
        }
        "fresh" => {
            if let Some(yes) = a0 {
                if yes == "y" || yes == "yes" {
                    store.reset_for_dim(384).await?;
                    app.turns.clear();
                    app.push_log("messages table recreated at VECTOR(384).");
                } else {
                    app.push_log("aborted (use :fresh y to confirm)");
                }
            } else {
                app.push_log("this erases all memories — confirm with :fresh y");
            }
        }
        "reset" => {
            if let Some(yes) = a1 {
                if yes == "y" || yes == "yes" {
                    store.truncate().await?;
                    app.push_log("store truncated.");
                } else {
                    app.push_log("aborted (use :reset y)");
                }
            } else {
                app.push_log("truncate store? confirm with :reset y");
            }
        }
        "clear" => {
            app.turns.clear();
            app.push_log("chat view cleared (memories kept).");
        }
        "profile" => {
            let p = store.profile(&app.agent).await?;
            app.push_log(format!(
                "agent={} total={} (private {} shared {}) novelty {:.2} salience {:.2}",
                p.agent_id, p.total, p.private, p.shared, p.avg_novelty, p.avg_salience
            ));
        }
        "" => {}
        other => app.push_log(format!("unknown command :{other} — :help")),
    }
    Ok(())
}

async fn do_seed(app: &mut App, store: &Arc<Store>) -> Result<()> {
    const SCRIPT: &[&str] = &[
        "I work at Google as a staff software engineer.",
        "We adopted a golden retriever puppy named Scout.",
        "My team standardized on jsonwebtoken v9 for auth.",
        "Actually, I rotated the JWT signing secret to v9 last sprint.",
        "Scout had his first vet appointment Tuesday and needs a booster shot in three weeks.",
        "I'm planning to move to Portland next month for a new role.",
        "The new role is a staff engineer position at a startup called Pinecone Labs.",
    ];
    app.push_log("seeding scripted memories…");
    let mut n = 0;
    for text in SCRIPT {
        let d = store.ingest(&app.agent, "private", "user", text).await
            .map_err(|e| dim_hint(e))?;
        if d.admitted { n += 1; }
    }
    app.push_log(format!("seeded {n}/{} memories — ask away", SCRIPT.len()));
    Ok(())
}

fn dim_hint(e: anyhow::Error) -> anyhow::Error {
    if e.to_string().contains("dimensions") {
        anyhow::anyhow!("{e} — run :fresh y to recreate the table at 384-dim")
    } else {
        e
    }
}

// ─── chat pipeline (background) ──────────────────────────────────────────────

fn send_chat(
    text: &str,
    app: &mut App,
    store: &Arc<Store>,
    openai_key: &Option<String>,
    tx: &tokio::sync::mpsc::UnboundedSender<UiEvent>,
) {
    let user_turn = Turn {
        role: Role::User,
        text: text.to_string(),
        meta: "ingesting…".into(),
        memories: vec![],
        pending: false,
    };
    let asst_turn = Turn {
        role: Role::Assistant,
        text: String::new(),
        meta: "recalling…".into(),
        memories: vec![],
        pending: true,
    };
    app.turns.push(user_turn);
    let asst_idx = app.turns.len();
    app.turns.push(asst_turn);
    let user_idx = asst_idx - 1;
    app.pending += 1;
    app.phase = Phase::Recalling;
    app.scroll = 0;

    let store = store.clone();
    let agent = app.agent.clone();
    let scope = app.scope.clone();
    let k = app.k;
    let mode = app.mode;
    let model = app.model.clone();
    let key = openai_key.clone();
    let history: Vec<(Role, String)> = app
        .turns
        .iter()
        .filter(|t| !t.pending && !t.text.is_empty())
        .cloned()
        .map(|t| (t.role, t.text))
        .collect::<Vec<_>>()
        .take_last(8);
    let text = text.to_string();
    let tx2 = tx.clone();

    tokio::spawn(async move {
        // 1. ingest the user message through the gate
        let decision = match store.ingest(&agent, &scope, "user", &text).await {
            Ok(d) => d,
            Err(e) => {
                let _ = tx2.send(UiEvent::Error(format!("ingest: {}", dim_hint(e))));
                return;
            }
        };
        let _ = tx2.send(UiEvent::UserIngested { turn: user_idx, decision });

        // 2. recall
        let hits = match store.search(&agent, &text, k).await {
            Ok(h) => h,
            Err(e) => {
                let _ = tx2.send(UiEvent::Error(format!("recall: {e}")));
                return;
            }
        };
        let _ = tx2.send(UiEvent::RecallDone { turn: asst_idx, hits: hits.clone() });

        // 3. answer (LLM grounded in memories + recent turns, or echo)
        let answer = if let Some(key) = &key {
            match answer_with_memory(key, &text, &hits, &history).await {
                Ok(a) => a,
                Err(e) => {
                    let _ = tx2.send(UiEvent::Error(format!("llm: {e}")));
                    fallback_answer(&hits)
                }
            }
        } else {
            fallback_answer(&hits)
        };

        // 4. ingest the assistant reply so the conversation has continuity
        let reply_id = match store.ingest(&agent, &scope, "assistant", &answer).await {
            Ok(d) => {
                if d.admitted { d.id } else { None }
            }
            Err(e) => {
                let _ = tx2.send(UiEvent::ReplyIngestError {
                    why: dim_hint(e).to_string(),
                });
                None
            }
        };

        // touch `mode`/`model` so the side panel can reflect eager consolidation
        let _ = (mode, model);

        let _ = tx2.send(UiEvent::AnswerDone { turn: asst_idx, text: answer, reply_id });
    });
}

fn spawn_consolidate(
    store: Arc<Store>,
    model: String,
    tx: tokio::sync::mpsc::UnboundedSender<UiEvent>,
) {
    tokio::spawn(async move {
        use consolidation::{ConsolidationWorker, ExtractiveConsolidator, OpenAiConsolidator};
        let model: std::sync::Arc<dyn consolidation::ConsolidationModel> = match model.as_str() {
            "openai" => {
                match std::env::var("OPENAI_API_KEY") {
                    Ok(k) => std::sync::Arc::from(Box::new(OpenAiConsolidator::new(k))
                        as Box<dyn consolidation::ConsolidationModel>),
                    Err(_) => {
                        let _ = tx.send(UiEvent::Error("OPENAI_API_KEY not set for --model openai".into()));
                        return;
                    }
                }
            }
            _ => std::sync::Arc::from(Box::new(ExtractiveConsolidator)
                as Box<dyn consolidation::ConsolidationModel>),
        };
        let before = store.pending_consolidation_count(model.name()).await.unwrap_or(-1);
        let worker = ConsolidationWorker {
            store: store.clone(),
            model,
            batch: 1000,
            interval_secs: 0,
        };
        match worker.run_once().await {
            Ok(n) => {
                let after = store.pending_consolidation_count("extractive").await.unwrap_or(-1);
                let _ = tx.send(UiEvent::Error(
                    format!("consolidated {n} memories · pending {before} → {after}")
                ));
            }
            Err(e) => {
                let _ = tx.send(UiEvent::Error(format!("consolidate: {e}")));
            }
        }
    });
}

// ─── LLM / answer ────────────────────────────────────────────────────────────

async fn answer_with_memory(
    key: &str,
    question: &str,
    hits: &[Hit],
    history: &[(Role, String)],
) -> Result<String> {
    let client = reqwest::Client::new();
    let memories = hits
        .iter()
        .enumerate()
        .map(|(i, h)| format!("({}) {}", i + 1, h.text))
        .collect::<Vec<_>>()
        .join("\n");

    let convo = history
        .iter()
        .map(|(r, t)| match r {
            Role::User => format!("User: {t}"),
            Role::Assistant => format!("Assistant: {t}"),
        })
        .collect::<Vec<_>>()
        .join("\n");

    let system = "You are a helpful assistant with long-term memory. \
        Answer the user's latest message using the provided memories when they are relevant. \
        If the memories do not contain the answer, say you don't have that in memory and answer generally. \
        Be concise and conversational.";
    let user = format!(
        "Relevant memories:\n{memories}\n\nConversation so far:\n{convo}\n\nUser: {question}\nAssistant:"
    );

    let req = json!({
        "model": "gpt-5-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    });
    let resp = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(key)
        .json(&req)
        .send()
        .await
        .context("OpenAI request failed")?;
    let status = resp.status();
    let body: Value = resp.json().await.context("parse OpenAI response")?;
    if !status.is_success() {
        anyhow::bail!("OpenAI {status}: {body}");
    }
    Ok(body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .trim()
        .to_string())
}

/// No-LLM fallback: surface the recalled memories so the memory pipeline is
/// still demonstrable without an API key.
fn fallback_answer(hits: &[Hit]) -> String {
    if hits.is_empty() {
        return "(no memories recalled — try :seed or tell me something first)".into();
    }
    let mut s = String::from("I don't have an LLM connected, but I recalled these memories:\n");
    for (i, h) in hits.iter().enumerate() {
        s.push_str(&format!("  [{}] (score {:.2}) {}\n", i + 1, h.score, h.text));
    }
    s
}

// ─── rendering ───────────────────────────────────────────────────────────────

fn render(f: &mut ratatui::Frame<'_>, app: &mut App) {
    let area = f.area();

    // outer vertical: [chat+panel | status | input]
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),
            Constraint::Length(1),
            Constraint::Length(3),
        ])
        .split(area);

    let body = chunks[0];
    let status_rect = chunks[1];
    let input_rect = chunks[2];

    // horizontal: [chat | memory panel]
    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(body);
    let chat_rect = body_chunks[0];
    let mem_rect = body_chunks[1];

    render_chat(f, app, chat_rect);
    render_memory_panel(f, app, mem_rect);
    render_status(f, app, status_rect);
    render_input(f, app, input_rect);

    if app.show_help {
        render_help(f, area);
    }
}

fn render_chat(f: &mut ratatui::Frame<'_>, app: &App, rect: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" chat · agent \"{}\" ", app.agent));
    let inner = block.inner(rect);
    f.render_widget(block, rect);

    let width = inner.width as usize;
    let mut lines: Vec<Line> = Vec::new();

    for turn in &app.turns {
        lines.extend(turn_lines(turn, width, app));
    }

    let total = lines.len();
    let h = inner.height as usize;
    let scroll = app.scroll.min(total.saturating_sub(h));
    let start = total.saturating_sub(h + scroll);
    let end = (start + h).min(total);
    let visible: Vec<Line> = lines[start.min(end)..end]
        .iter()
        .cloned()
        .collect();
    // avoid empty-slice panic when nothing to show
    let visible = if visible.is_empty() { vec![Line::from("")] } else { visible };

    let p = Paragraph::new(visible).wrap(Wrap { trim: false });
    f.render_widget(p, inner);
}

fn turn_lines(turn: &Turn, width: usize, app: &App) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    let label = match turn.role {
        Role::User => Span::styled(" you ", Style::default().bg(Color::Cyan).fg(Color::Black).add_modifier(Modifier::BOLD)),
        Role::Assistant => Span::styled(" assistant ", Style::default().bg(Color::Green).fg(Color::Black).add_modifier(Modifier::BOLD)),
    };
    let meta = Span::styled(format!("  {}", turn.meta), Style::default().fg(Color::DarkGray));
    out.push(Line::from(vec![label, meta]));

    let body_style = match turn.role {
        Role::User => Style::default().fg(Color::White),
        Role::Assistant => Style::default().fg(Color::Gray),
    };

    if turn.pending {
        let spin = ['|', '/', '-', '\\'][app.spinner];
        out.push(Line::from(vec![
            Span::styled(format!("  {spin} thinking…"), Style::default().fg(Color::DarkGray)),
        ]));
    } else {
        for l in wrap_text(&turn.text, width.saturating_sub(2)) {
            out.push(Line::from(vec![Span::styled(format!("  {l}"), body_style)]));
        }
        if !turn.memories.is_empty() && matches!(turn.role, Role::Assistant) {
            let ids: Vec<String> = turn.memories.iter()
                .map(|h| format!("[{}]·{:.2}", h.id, h.score))
                .collect();
            out.push(Line::from(vec![Span::styled(
                format!("  memories: {}", ids.join("  ")),
                Style::default().fg(Color::DarkGray),
            )]));
        }
    }
    out.push(Line::from(""));
    out
}

fn render_memory_panel(f: &mut ratatui::Frame<'_>, app: &App, rect: Rect) {
    let phase_label = match app.phase {
        Phase::Idle => "idle",
        Phase::Recalling => "recalling…",
        Phase::Thinking => "thinking…",
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" recalled memory · k={} · {} ", app.k, phase_label));
    let inner = block.inner(rect);
    f.render_widget(block, rect);

    // most recent assistant turn with memories
    let last = app.turns.iter().rev().find(|t| !t.memories.is_empty());
    let items: Vec<ListItem> = match last {
        Some(t) => t.memories.iter().map(|h| {
            let head = Line::from(vec![
                Span::styled(format!("[{}] ", h.id), Style::default().fg(Color::Yellow)),
                Span::styled(format!("{:.2}", h.score), Style::default().fg(Color::DarkGray)),
            ]);
            let body = wrap_text(&h.text, inner.width as usize);
            let mut lines = vec![head];
            for l in body.iter().take(3) {
                lines.push(Line::from(Span::styled(format!("  {l}"), Style::default().fg(Color::Gray))));
            }
            ListItem::new(lines)
        }).collect(),
        None => vec![ListItem::new(Line::from(Span::styled(
            "no memories recalled yet — send a message or :seed",
            Style::default().fg(Color::DarkGray),
        )))],
    };
    let list = List::new(items);
    f.render_widget(list, inner);
}

fn render_status(f: &mut ratatui::Frame<'_>, app: &App, rect: Rect) {
    let llm = if app.llm_on { "llm:on" } else { "llm:echo" };
    let mode = match app.mode {
        ConsolidationMode::None => "none",
        ConsolidationMode::Lazy => "lazy",
        ConsolidationMode::Eager => "eager",
    };
    let left = format!(
        " agent={} · scope={} · mode={} · cmodel={} · k={} · {} ",
        app.agent, app.scope, mode, app.model, app.k, llm,
    );
    let log = if app.log.is_empty() { String::new() } else { app.log.clone() };
    let line = Line::from(vec![
        Span::styled(left, Style::default().fg(Color::Black).bg(Color::DarkGray)),
        Span::raw(" "),
        Span::styled(log, Style::default().fg(Color::DarkGray)),
    ]);
    f.render_widget(Paragraph::new(line), rect);
}

fn render_input(f: &mut ratatui::Frame<'_>, app: &App, rect: Rect) {
    let block = Block::default().borders(Borders::ALL).title(" message (:help · ↑/↓ scroll · ? help · Esc quit) ");
    f.render_widget(block, rect);

    let inner = rect; // borders already drawn; place text manually
    let prompt = "› ";
    let line = Line::from(vec![
        Span::styled(prompt, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(app.input.as_str()),
    ]);
    let p = Paragraph::new(line).wrap(Wrap { trim: false });
    // render inside the bordered area
    let text_rect = Rect { x: rect.x + 1, y: rect.y + 1, width: rect.width.saturating_sub(2), height: 1 };
    f.render_widget(p, text_rect);

    let prompt_w = prompt.chars().count() as u16;
    let col = char_count(&app.input[..app.cursor]) as u16;
    f.set_cursor_position((text_rect.x + prompt_w + col, text_rect.y));
    let _ = inner;
}

fn render_help(f: &mut ratatui::Frame<'_>, area: Rect) {
    let w = 64.min(area.width);
    let h = 22.min(area.height);
    let x = area.x + (area.width - w) / 2;
    let y = area.y + (area.height - h) / 2;
    let rect = Rect { x, y, width: w, height: h };
    f.render_widget(Clear, rect);
    let help = "\
memchat — chat with a memory-grounded agent

typing
  <text>        send a message (ingested → recalled → answered)
  :<command>    run a command (see below)
  ↑ / ↓         scroll transcript   PgUp/PgDn · Home/End
  Esc / Ctrl-C  quit
  ?             toggle this help

commands
  :seed            load a scripted demo conversation
  :agent <id>      switch agent (multi-agent isolation)
  :scope private|shared
  :mode none|lazy|eager
  :model extractive|openai
  :k <n>           recall top-k
  :consolidate [m] run consolidation worker once
  :profile         per-agent memory stats
  :fresh y         recreate messages table at 384-dim (erases all)
  :reset y         truncate the store
  :clear           clear the chat view (keeps memories)

each turn: gate decides admit/drop → recall top-k → LLM grounds reply
in recalled memories → reply ingested as assistant memory (continuity).
";
    let p = Paragraph::new(help)
        .style(Style::default().fg(Color::White))
        .block(Block::default().borders(Borders::ALL).title(" help · press ? to close "));
    f.render_widget(p, rect);
}

// ─── text helpers ────────────────────────────────────────────────────────────

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out: Vec<String> = Vec::new();
    for paragraph in text.split('\n') {
        let words: Vec<&str> = paragraph.split_whitespace().collect();
        if words.is_empty() {
            out.push(String::new());
            continue;
        }
        let mut line = String::new();
        for w in words {
            let wl = w.chars().count();
            let cur = line.chars().count();
            if cur == 0 {
                line.push_str(w);
            } else if cur + 1 + wl > width {
                out.push(std::mem::take(&mut line));
                line.push_str(w);
            } else {
                line.push(' ');
                line.push_str(w);
            }
        }
        if !line.is_empty() {
            out.push(line);
        }
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

fn char_count(s: &str) -> usize {
    s.chars().count()
}

fn prev_char_boundary(s: &str, i: usize) -> Option<usize> {
    if i == 0 { return None; }
    let mut j = i - 1;
    while j > 0 && !s.is_char_boundary(j) { j -= 1; }
    Some(j)
}

fn next_char_boundary(s: &str, i: usize) -> Option<usize> {
    if i >= s.len() { return None; }
    let mut j = i + 1;
    while j < s.len() && !s.is_char_boundary(j) { j += 1; }
    Some(j)
}

// TakeLast for Vec — small helper since std doesn't have it on Vec.
trait TakeLastExt<T> {
    fn take_last(self, n: usize) -> Vec<T>;
}
impl<T: Clone> TakeLastExt<T> for Vec<T> {
    fn take_last(self, n: usize) -> Vec<T> {
        let start = self.len().saturating_sub(n);
        self[start..].to_vec()
    }
}
