//! memplay — interactive developer harness for the memory substrate.
//!
//! A REPL that showcases the full pipeline live: type a message and watch the
//! LLM-free gate decide (novelty + salience + correction) in real time; ask a
//! question and see the BM25 + dense + RRF + cross-encoder hits, then flip
//! between raw / lazy-consolidated / eager-consolidated views of the same
//! recall to compare what consolidation does to memory.
//!
//! Run:  cargo run --bin memplay -- [--agent <id>] [--config <path>]
//!
//! Inside the REPL:
//!   <text>           ingest through the gate (any line not starting with : or ?)
//!   ?<query>         recall top-k (uses current :mode and :model)
//!   :help            show this list
//!   :seed            load a scripted conversation that exercises the gate
//!   :list [n]        recent memories, newest first
//!   :show <id>       raw memory + any stored consolidation
//!   :consolidate     run the consolidation worker once over pending memories
//!   :profile         per-agent memory stats
//!   :mode <m>        recall consolidation mode: none | lazy | eager
//!   :model <m>       consolidation model: extractive | openai
//!   :scope <s>       ingest scope for new memories: private | shared
//!   :role <r>        ingest role: user | assistant | system
//!   :agent <id>      switch active agent (multi-agent isolation demo)
//!   :k <n>           recall top-k
//!   :count           total rows in the store
//!   :reset           truncate the store (confirms first)
//!   :quit | :q       exit

use anyhow::{Context, Result};
use clap::Parser;
use consolidation::{
    ConsolidationMode, ConsolidationModel, ConsolidationWorker, ExtractiveConsolidator,
    OpenAiConsolidator,
};
use std::io::{self, BufRead, BufReader, Write};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use store::{Config, Store};

#[derive(Parser)]
#[command(name = "memplay", about = "Interactive memory substrate harness", version)]
struct Cli {
    #[arg(long, default_value = "memoryd.toml")]
    config: String,
    #[arg(long, default_value = "dev")]
    agent: String,
    /// Drop and recreate the messages table at 384-dim (bge-small) for a clean
    /// slate. Use this if a prior locomo_qa / longmemeval_qa run left the table
    /// at a different vector width.
    #[arg(long, default_value_t = false)]
    fresh: bool,
}

struct Repl {
    store: Arc<Store>,
    agent: String,
    scope: String,
    role: String,
    mode: ConsolidationMode,
    model_name: String,
    k: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load(&cli.config).map_err(|e| {
        anyhow::anyhow!(
            "failed to load config '{}': {e}\n\
             Make sure memoryd.toml exists (see README for the template).",
            cli.config
        )
    })?;
    let store = Arc::new(Store::connect(cfg).await?);

    if cli.fresh {
        store.reset_for_dim(384).await?;
        eprintln!("[memplay] --fresh: reset messages table to VECTOR(384)");
    }

    let mut repl = Repl {
        store,
        agent: cli.agent,
        scope: "private".into(),
        role: "user".into(),
        mode: ConsolidationMode::None,
        model_name: "extractive".into(),
        k: 5,
    };

    println!("memplay — memory substrate harness");
    println!("store: {} messages", repl.store.count().await?);
    println!("agent=\"{}\" scope={} role={} mode={} model={} k={}",
        repl.agent, repl.scope, repl.role, mode_str(repl.mode), repl.model_name, repl.k);
    println!("type :help for commands, :seed to load a demo, or just start typing.");

    let stdin = io::stdin();
    let mut input = BufReader::new(stdin.lock());
    loop {
        print!("\n[{} {}] > ", repl.agent, repl.scope);
        io::stdout().flush().ok();
        let mut line = String::new();
        if input.read_line(&mut line).unwrap_or(0) == 0 {
            break; // EOF
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let (kind, rest) = split_prefix(&line);
        let quit = match kind {
            Prefix::Command => dispatch_cmd(&mut repl, &mut input, rest).await?,
            Prefix::Query => {
                if let Err(e) = do_recall(&mut repl, rest).await {
                    eprintln!("recall error: {e:#}");
                }
                false
            }
            Prefix::Ingest => {
                if let Err(e) = do_ingest(&mut repl, &line).await {
                    eprintln!("ingest error: {e:#}");
                }
                false
            }
        };
        if quit {
            break;
        }
    }
    println!("\nbye.");
    Ok(())
}

// ─── prefix parsing ──────────────────────────────────────────────────────────

enum Prefix { Command, Query, Ingest }

fn split_prefix(line: &str) -> (Prefix, &str) {
    if let Some(rest) = line.strip_prefix(':') {
        return (Prefix::Command, rest);
    }
    if let Some(rest) = line.strip_prefix('?') {
        return (Prefix::Query, rest);
    }
    (Prefix::Ingest, line)
}

// ─── command dispatch ────────────────────────────────────────────────────────

/// Returns `true` when the REPL should exit.
async fn dispatch_cmd(repl: &mut Repl, input: &mut BufReader<io::StdinLock<'_>>, raw: &str) -> Result<bool> {
    let mut parts = raw.split_whitespace();
    let cmd = parts.next().unwrap_or("");
    let arg0 = parts.next();
    let _arg1 = parts.next();
    match cmd {
        "" => Ok(false),
        "help" | "h" | "?" => {
            print_help();
            Ok(false)
        }
        "quit" | "q" | "exit" => Ok(true),
        "seed" => {
            do_seed(repl).await?;
            Ok(false)
        }
        "list" | "ls" => {
            let n = arg0.and_then(|s| s.parse::<i64>().ok()).unwrap_or(10);
            do_list(repl, n).await?;
            Ok(false)
        }
        "show" => {
            let id = arg0.and_then(|s| s.parse::<i64>().ok()).context("usage: :show <id>")?;
            do_show(repl, id).await?;
            Ok(false)
        }
        "consolidate" | "cons" => {
            if let Some(m) = arg0 {
                repl.model_name = m.to_string();
            }
            do_consolidate(repl).await?;
            Ok(false)
        }
        "profile" | "p" => {
            do_profile(repl).await?;
            Ok(false)
        }
        "mode" => {
            let m = arg0.context("usage: :mode none|lazy|eager")?;
            repl.mode = m.parse().context("bad mode")?;
            println!("mode → {}", mode_str(repl.mode));
            Ok(false)
        }
        "model" => {
            let m = arg0.context("usage: :model extractive|openai")?;
            make_model(m)?; // validate early so typos don't surface at recall
            repl.model_name = m.to_string();
            println!("model → {m}");
            Ok(false)
        }
        "scope" => {
            let s = arg0.context("usage: :scope private|shared")?;
            anyhow::ensure!(s == "private" || s == "shared", "scope must be private or shared");
            repl.scope = s.to_string();
            println!("scope → {}", repl.scope);
            Ok(false)
        }
        "role" => {
            let r = arg0.context("usage: :role user|assistant|system")?;
            anyhow::ensure!(
                r == "user" || r == "assistant" || r == "system",
                "role must be user, assistant, or system"
            );
            repl.role = r.to_string();
            println!("role → {}", repl.role);
            Ok(false)
        }
        "agent" => {
            let a = arg0.context("usage: :agent <id>")?;
            repl.agent = a.to_string();
            println!("agent → {}", repl.agent);
            Ok(false)
        }
        "k" => {
            let n = arg0.and_then(|s| s.parse::<usize>().ok()).context("usage: :k <n>")?;
            repl.k = n;
            println!("k → {n}");
            Ok(false)
        }
        "count" => {
            println!("total rows: {}", repl.store.count().await?);
            Ok(false)
        }
        "fresh" => {
            confirm(input, "drop + recreate messages table at 384-dim (erases all memories)?")?;
            repl.store.reset_for_dim(384).await?;
            println!("messages table recreated at VECTOR(384); 0 rows.");
            Ok(false)
        }
        "reset" => {
            confirm(input, "truncate all messages?")?;
            repl.store.truncate().await?;
            println!("store truncated.");
            Ok(false)
        }
        other => {
            eprintln!("unknown command ':{other}' — type :help");
            Ok(false)
        }
    }
}

fn print_help() {
    let help = "\
<text>            ingest through the novelty + salience gate
?<query>          recall top-k (uses current :mode and :model)
:seed             load a scripted demo conversation
:list [n]         recent memories (default 10)
:show <id>        raw memory + stored consolidation
:consolidate [m]  run consolidation worker once over pending memories
:profile          per-agent memory stats
:mode <m>         none | lazy | eager   (recall view)
:model <m>        extractive | openai   (consolidation model)
:scope <s>        private | shared      (ingest scope)
:role <r>         user | assistant | system
:agent <id>       switch active agent (multi-agent isolation)
:k <n>            recall top-k
:count            total rows in the store
:fresh            drop + recreate messages table at 384-dim (erases everything)
:reset            truncate the store (keeps vector dim)
:quit             exit";
    println!("{help}");
}

// ─── actions ─────────────────────────────────────────────────────────────────

async fn do_ingest(repl: &mut Repl, text: &str) -> Result<()> {
    let d = repl.store.ingest(&repl.agent, &repl.scope, &repl.role, text).await.map_err(|e| {
        if e.to_string().contains("dimensions") {
            anyhow::anyhow!(
                "{e}\n  hint: vector dim mismatch — run :fresh (or relaunch with --fresh) \
                 to recreate the table at 384-dim"
            )
        } else {
            e
        }
    })?;
    if d.admitted {
        println!(
            "  ✓ stored   id={:<5} novelty={:.3}  salience={:.3}  [{}ms]  {}",
            d.id.unwrap(),
            d.novelty,
            d.salience,
            d.timings.total_ms as u64,
            d.reason,
        );
    } else {
        println!(
            "  ✗ dropped  novelty={:.3}  salience={:.3}  reason=\"{}\"",
            d.novelty, d.salience, d.reason
        );
        if d.correction {
            println!("    (flagged as correction — would bypass, but not admitted)");
        }
    }
    Ok(())
}

async fn do_recall(repl: &mut Repl, query: &str) -> Result<()> {
    let query = query.trim();
    if query.is_empty() {
        eprintln!("usage: ?<query>");
        return Ok(());
    }
    let hits = repl.store.search(&repl.agent, query, repl.k).await?;
    println!(
        "recall  q=\"{query}\"  k={}  mode={}  model={}  → {} hit(s)",
        repl.k,
        mode_str(repl.mode),
        repl.model_name,
        hits.len(),
    );
    if hits.is_empty() {
        return Ok(());
    }

    match repl.mode {
        ConsolidationMode::None => {
            for (i, h) in hits.iter().enumerate() {
                println!("  [{}] id={}  score={:.4}", i + 1, h.id, h.score);
                println!("      {}", wrap_indent(&h.text, 8));
            }
        }
        ConsolidationMode::Lazy => {
            let model = make_model(&repl.model_name)?;
            for (i, h) in hits.iter().enumerate() {
                let refs = vec![h.text.as_str()];
                let c = model.consolidate(&refs).await
                    .with_context(|| format!("lazy consolidation failed for id={}", h.id))?;
                println!("  [{}] id={}  score={:.4}  topic={}", i + 1, h.id, h.score, c.topic_path);
                println!("      title: {}", c.title);
                println!("      body:  {}", wrap_indent(&c.body, 8));
                if !c.foresight.is_empty() {
                    for f in &c.foresight {
                        let exp = f.expires.as_deref().unwrap_or("durable");
                        println!("      → {}  [{}]", f.statement, exp);
                    }
                }
            }
        }
        ConsolidationMode::Eager => {
            for (i, h) in hits.iter().enumerate() {
                let c = repl.store.get_consolidation(h.id, &repl.model_name).await?;
                println!("  [{}] id={}  score={:.4}", i + 1, h.id, h.score);
                match c {
                    Some((topic, title, body, foresight_json)) => {
                        println!("      [{}] {}", topic, title);
                        println!("      {}", wrap_indent(&body, 8));
                        print_foresight_json(&foresight_json, 8);
                    }
                    None => {
                        println!("      {} (not consolidated — run :consolidate)", wrap_indent(&h.text, 8));
                    }
                }
            }
        }
    }
    Ok(())
}

async fn do_list(repl: &mut Repl, n: i64) -> Result<()> {
    let rows = repl.store.list(&repl.agent, n).await?;
    if rows.is_empty() {
        println!("no memories for agent '{}'", repl.agent);
        return Ok(());
    }
    for r in &rows {
        let age = format_ts(r.ts);
        let nov = r.novelty.map_or("-".to_string(), |v| format!("{v:.2}"));
        let sal = r.salience.map_or("-".to_string(), |v| format!("{v:.2}"));
        println!("  id={:<5} {} n={} s={} [{}]", r.id, age, nov, sal, r.scope);
        println!("      {}", wrap_indent(&r.text, 8));
    }
    Ok(())
}

async fn do_show(repl: &mut Repl, id: i64) -> Result<()> {
    let rows = repl.store.list(&repl.agent, i64::MAX).await?;
    let row = rows.iter().find(|r| r.id == id);
    let Some(r) = row else {
        println!("no memory id={id} for agent '{}'", repl.agent);
        return Ok(());
    };
    let nov = r.novelty.map_or("-".to_string(), |v| format!("{v:.3}"));
    let sal = r.salience.map_or("-".to_string(), |v| format!("{v:.3}"));
    println!("id={}  agent={}  scope={}  role={}  ts={}", r.id, repl.agent, r.scope, r.role, format_ts(r.ts));
    println!("novelty={}  salience={}", nov, sal);
    println!("raw:");
    println!("  {}", wrap_indent(&r.text, 4));

    let c = repl.store.get_consolidation(id, &repl.model_name).await?;
    match c {
        Some((topic, title, body, foresight_json)) => {
            println!("consolidation [{}]:", repl.model_name);
            println!("  topic: {}", topic);
            println!("  title: {}", title);
            println!("  body:  {}", wrap_indent(&body, 8));
            print_foresight_json(&foresight_json, 8);
        }
        None => println!("consolidation [{}]: <none — run :consolidate>", repl.model_name),
    }
    Ok(())
}

async fn do_consolidate(repl: &mut Repl) -> Result<()> {
    let pending_before = repl.store.pending_consolidation_count(&repl.model_name).await?;
    let model = make_model(&repl.model_name)?;
    let worker = ConsolidationWorker {
        store: repl.store.clone(),
        model: Arc::from(model),
        batch: 1000,
        interval_secs: 0,
    };
    let n = worker.run_once().await?;
    let pending_after = repl.store.pending_consolidation_count(&repl.model_name).await?;
    println!(
        "consolidated {n} memories (model={})  pending: {} → {}",
        repl.model_name, pending_before, pending_after,
    );
    Ok(())
}

async fn do_profile(repl: &mut Repl) -> Result<()> {
    let p = repl.store.profile(&repl.agent).await?;
    println!("agent:    {}", p.agent_id);
    println!("total:    {}", p.total);
    println!("  private: {}", p.private);
    println!("  shared:  {}", p.shared);
    if p.total > 0 {
        println!("novelty:  {:.3}  (avg)", p.avg_novelty);
        println!("salience: {:.3}  (avg)", p.avg_salience);
    }
    if let (Some(oldest), Some(newest)) = (p.oldest_ts, p.newest_ts) {
        println!("span:     {} → {}", format_ts(oldest), format_ts(newest));
    }
    Ok(())
}

/// A scripted conversation that exercises every gate branch: novel topics,
/// near-duplicates dropped by low novelty, noise dropped by the salience floor,
/// and a correction that bypasses the gate.
async fn do_seed(repl: &mut Repl) -> Result<()> {
    const SCRIPT: &[&str] = &[
        "I work at Google as a staff software engineer.",
        "We adopted a golden retriever puppy named Scout.",
        "My team standardized on jsonwebtoken v9 for auth.",
        "ok",
        "I work at Google as a staff software engineer.",
        "Actually, I rotated the JWT signing secret to v9 last sprint.",
        "thanks",
        "Scout had his first vet appointment Tuesday and needs a booster shot in three weeks.",
        "I'm planning to move to Portland next month for a new role.",
        "Portland rent is around 2200 a month for a one-bedroom.",
        "The new role is a staff engineer position at a startup called Pinecone Labs.",
    ];
    println!("seeding {} scripted messages through the gate…", SCRIPT.len());
    println!("  {:<3} {:<6} {:>7} {:>8}  {:<22} message", "#", "admit", "novelty", "salience", "reason");
    let mut admitted = 0usize;
    for (i, text) in SCRIPT.iter().enumerate() {
        let d = repl.store.ingest(&repl.agent, "private", "user", text).await.map_err(|e| {
            if e.to_string().contains("dimensions") {
                anyhow::anyhow!(
                    "{e}\n  hint: vector dim mismatch — run :fresh (or relaunch with --fresh) \
                     to recreate the table at 384-dim"
                )
            } else {
                e
            }
        })?;
        println!(
            "  {:<3} {:<6} {:>7.3} {:>8.3}  {:<22} {}",
            i + 1,
            if d.admitted { "ADMIT" } else { "drop" },
            d.novelty,
            d.salience,
            d.reason,
            truncate(text, 48),
        );
        if d.admitted {
            admitted += 1;
        }
    }
    println!("seeded: {admitted}/{} admitted. Try: ?where does the user work   then :mode eager", SCRIPT.len());
    Ok(())
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn make_model(name: &str) -> Result<Box<dyn ConsolidationModel>> {
    match name {
        "extractive" => Ok(Box::new(ExtractiveConsolidator)),
        "openai" => {
            let key = std::env::var("OPENAI_API_KEY")
                .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set; required for --model openai"))?;
            Ok(Box::new(OpenAiConsolidator::new(key)))
        }
        other => anyhow::bail!("unknown model '{other}': use extractive or openai"),
    }
}

fn mode_str(m: ConsolidationMode) -> &'static str {
    match m {
        ConsolidationMode::None => "none",
        ConsolidationMode::Lazy => "lazy",
        ConsolidationMode::Eager => "eager",
    }
}

fn format_ts(millis: i64) -> String {
    let then = UNIX_EPOCH + Duration::from_millis(millis.max(0) as u64);
    let elapsed = SystemTime::now().duration_since(then).unwrap_or_default().as_secs();
    match elapsed {
        s if s < 60 => format!("{s}s ago"),
        s if s < 3600 => format!("{}m ago", s / 60),
        s if s < 86400 => format!("{}h ago", s / 3600),
        s => format!("{}d ago", s / 86400),
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

/// Wrap long text to ~80 cols and indent every continuation line by `indent`
/// spaces so multi-line bodies line up under their label.
fn wrap_indent(text: &str, indent: usize) -> String {
    const WIDTH: usize = 80;
    let pad = " ".repeat(indent);
    let mut out = String::new();
    for (li, paragraph) in text.split('\n').enumerate() {
        if li > 0 {
            out.push('\n');
            out.push_str(&pad);
        }
        let words: Vec<&str> = paragraph.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }
        let mut col = 0;
        for w in words {
            let need = w.chars().count() + if col == 0 { 0 } else { 1 };
            if col + need > WIDTH && col > 0 {
                out.push('\n');
                out.push_str(&pad);
                col = 0;
            }
            if col > 0 {
                out.push(' ');
                col += 1;
            }
            out.push_str(w);
            col += w.chars().count();
        }
    }
    out
}

fn print_foresight_json(json: &str, indent: usize) {
    let pad = " ".repeat(indent);
    match serde_json::from_str::<Vec<serde_json::Value>>(json) {
        Ok(arr) if !arr.is_empty() => {
            for item in arr {
                let stmt = item.get("statement").and_then(|v| v.as_str()).unwrap_or("?");
                let exp = item.get("expires").and_then(|v| v.as_str()).unwrap_or("durable");
                println!("{pad}→ {stmt}  [{exp}]");
            }
        }
        Ok(_) => {}
        Err(_) => println!("{pad}(foresight: {json})"),
    }
}

fn confirm(input: &mut BufReader<io::StdinLock<'_>>, prompt: &str) -> Result<()> {
    print!("{prompt} [y/N] ");
    io::stdout().flush().ok();
    let mut s = String::new();
    if input.read_line(&mut s)? == 0 {
        anyhow::bail!("EOF");
    }
    if !s.trim().eq_ignore_ascii_case("y") {
        anyhow::bail!("aborted");
    }
    Ok(())
}
