//! memoryd CLI — interact with a local memory store.
//!
//! Subcommands:
//!   remember <text> [--scope private|shared] [--role user|assistant|system]
//!   recall   <query> [-k N] [--mode none|lazy|eager] [--consolidation-model extractive|openai]
//!   list     [-n N]
//!   promote  <id>
//!   forget   <id>
//!   profile
//!   consolidate [--model extractive|openai] [--batch N]   (run once)
//!   worker      [--model extractive|openai] [--interval N] (run forever)

use anyhow::Result;
use clap::{Parser, Subcommand};
use consolidation::{
    ConsolidationMode, ConsolidationWorker, ExtractiveConsolidator, OpenAiConsolidator,
};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use store::{Config, Store};

#[derive(Parser)]
#[command(
    name = "memoryd",
    about = "Local-first sovereign memory substrate",
    version
)]
struct Cli {
    #[arg(long, default_value = "memoryd.toml")]
    config: String,
    #[arg(long, default_value = "default")]
    agent: String,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Store a memory (runs through the novelty + salience gate)
    Remember {
        text: String,
        #[arg(long, default_value = "private")]
        scope: String,
        #[arg(long, default_value = "user")]
        role: String,
    },
    /// Retrieve top-k memories (none / lazy / eager consolidation modes)
    Recall {
        query: String,
        #[arg(short, long, default_value_t = 5)]
        k: usize,
        /// Consolidation mode: none (raw), lazy (consolidate now), eager (use stored)
        #[arg(long, default_value = "none")]
        mode: String,
        /// Which consolidation model to use for lazy mode
        #[arg(long, default_value = "extractive")]
        consolidation_model: String,
    },
    /// Show recent memories, newest first
    List {
        #[arg(short, long, default_value_t = 10)]
        n: i64,
    },
    /// Elevate a private memory to shared scope
    Promote { id: i64 },
    /// Delete a memory permanently
    Forget { id: i64 },
    /// Show memory statistics for the agent
    Profile,
    /// Run consolidation once for all pending memories
    Consolidate {
        #[arg(long, default_value = "extractive")]
        model: String,
        #[arg(long, default_value_t = 50)]
        batch: i64,
    },
    /// Run the eager consolidation worker loop
    Worker {
        #[arg(long, default_value = "extractive")]
        model: String,
        /// Seconds between ticks
        #[arg(long, default_value_t = 30)]
        interval: u64,
        #[arg(long, default_value_t = 50)]
        batch: i64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let cfg = Config::load(&cli.config).map_err(|e| {
        anyhow::anyhow!(
            "failed to load config '{}': {}\n\
             Make sure memoryd.toml exists (see README for the template).",
            cli.config,
            e
        )
    })?;

    let store = Arc::new(Store::connect(cfg).await?);

    match cli.cmd {
        Cmd::Remember { text, scope, role } => {
            let d = store.ingest(&cli.agent, &scope, &role, &text).await?;
            if d.admitted {
                println!(
                    "stored   id={:<6}  novelty={:.3}  salience={:.3}",
                    d.id.unwrap(),
                    d.novelty,
                    d.salience
                );
            } else {
                println!(
                    "dropped  reason={}  novelty={:.3}  salience={:.3}",
                    d.reason, d.novelty, d.salience
                );
            }
        }

        Cmd::Recall {
            query,
            k,
            mode,
            consolidation_model,
        } => {
            let mode: ConsolidationMode = mode.parse()?;
            let hits = store.search(&cli.agent, &query, k).await?;

            if hits.is_empty() {
                println!("no results");
                return Ok(());
            }

            match mode {
                ConsolidationMode::None => {
                    for (i, h) in hits.iter().enumerate() {
                        println!("[{}] id={}  score={:.4}", i + 1, h.id, h.score);
                        println!("    {}", h.text);
                    }
                }
                ConsolidationMode::Lazy => {
                    let model = make_model(&consolidation_model)?;
                    for (i, h) in hits.iter().enumerate() {
                        let refs = vec![h.text.as_str()];
                        let c = model.consolidate(&refs).await?;
                        println!(
                            "[{}] id={}  score={:.4}  topic={}",
                            i + 1,
                            h.id,
                            h.score,
                            c.topic_path
                        );
                        println!("    title: {}", c.title);
                        println!("    body:  {}", c.body);
                        if !c.foresight.is_empty() {
                            for f in &c.foresight {
                                let exp = f.expires.as_deref().unwrap_or("durable");
                                println!("    → {} [{}]", f.statement, exp);
                            }
                        }
                    }
                }
                ConsolidationMode::Eager => {
                    // Use pre-consolidated body from the DB when available.
                    for (i, h) in hits.iter().enumerate() {
                        let c = store
                            .get_consolidation(h.id, &consolidation_model)
                            .await?;
                        println!("[{}] id={}  score={:.4}", i + 1, h.id, h.score);
                        if let Some((topic, title, body, _foresight)) = c {
                            println!("    [{}] {}", topic, title);
                            println!("    {}", body);
                        } else {
                            println!("    {} (not consolidated)", h.text);
                        }
                    }
                }
            }
        }

        Cmd::List { n } => {
            let rows = store.list(&cli.agent, n).await?;
            if rows.is_empty() {
                println!("no memories for agent '{}'", cli.agent);
            } else {
                for r in &rows {
                    let age = format_ts(r.ts);
                    let nov = r.novelty.map_or("-".into(), |v| format!("{v:.2}"));
                    let sal = r.salience.map_or("-".into(), |v| format!("{v:.2}"));
                    println!(
                        "id={:<6}  {}  n={} s={}  [{}]  {}",
                        r.id, age, nov, sal, r.scope, r.text
                    );
                }
            }
        }

        Cmd::Promote { id } => {
            if store.promote(&cli.agent, id).await? {
                println!("promoted id={id} → shared");
            } else {
                println!("not found: id={id} (must belong to --agent '{}')", cli.agent);
            }
        }

        Cmd::Forget { id } => {
            if store.forget(&cli.agent, id).await? {
                println!("deleted id={id}");
            } else {
                println!("not found: id={id} (must belong to --agent '{}')", cli.agent);
            }
        }

        Cmd::Profile => {
            let p = store.profile(&cli.agent).await?;
            println!("agent:     {}", p.agent_id);
            println!("total:     {}", p.total);
            println!("  private: {}", p.private);
            println!("  shared:  {}", p.shared);
            if p.total > 0 {
                println!("novelty:   {:.3}  (avg)", p.avg_novelty);
                println!("salience:  {:.3}  (avg)", p.avg_salience);
            }
            if let (Some(oldest), Some(newest)) = (p.oldest_ts, p.newest_ts) {
                println!("oldest:    {}", format_ts(oldest));
                println!("newest:    {}", format_ts(newest));
            }
        }

        Cmd::Consolidate { model, batch } => {
            let m = make_model(&model)?;
            let worker = ConsolidationWorker {
                store,
                model: Arc::from(m),
                batch,
                interval_secs: 0,
            };
            let n = worker.run_once().await?;
            println!("consolidated {n} memories (model={})", model);
        }

        Cmd::Worker {
            model,
            interval,
            batch,
        } => {
            let m = make_model(&model)?;
            println!(
                "starting consolidation worker  model={}  interval={interval}s  batch={batch}",
                model
            );
            let worker = ConsolidationWorker {
                store,
                model: Arc::from(m),
                batch,
                interval_secs: interval,
            };
            worker.run_loop().await;
        }
    }

    Ok(())
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn make_model(
    name: &str,
) -> Result<Box<dyn consolidation::ConsolidationModel>> {
    match name {
        "extractive" => Ok(Box::new(ExtractiveConsolidator)),
        "openai" => {
            let key = std::env::var("OPENAI_API_KEY").map_err(|_| {
                anyhow::anyhow!("OPENAI_API_KEY not set; required for --model openai")
            })?;
            Ok(Box::new(OpenAiConsolidator::new(key)))
        }
        other => anyhow::bail!("unknown model '{other}': use extractive or openai"),
    }
}

fn format_ts(millis: i64) -> String {
    let then = UNIX_EPOCH + Duration::from_millis(millis as u64);
    let elapsed = SystemTime::now()
        .duration_since(then)
        .unwrap_or_default()
        .as_secs();
    match elapsed {
        s if s < 60 => format!("{s}s ago   "),
        s if s < 3600 => format!("{}m ago  ", s / 60),
        s if s < 86400 => format!("{}h ago  ", s / 3600),
        s => format!("{}d ago  ", s / 86400),
    }
}
