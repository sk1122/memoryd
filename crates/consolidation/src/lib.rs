//! M5: Consolidation worker — EverMemOS-style background enrichment.
//!
//! Takes raw admitted memories and distils them into durable MemCell fields:
//!   topic_path  — hierarchical topic label  (e.g. "work/engineering")
//!   title       — concise noun phrase
//!   body        — enriched prose (key facts, deduped)
//!   foresight   — predicted future-relevant statements with optional expiry
//!
//! Two implementations ship:
//!   ExtractiveConsolidator — heuristic, local, zero extra deps
//!   OpenAiConsolidator     — LLM, structured JSON output (matches spike_llm.rs)
//!
//! Gemma-3-4B via ort is tracked as LocalConsolidator (stub — ONNX export
//! pending; swap in by implementing the trait on it).
//!
//! The worker operates in two modes controlled by the caller:
//!   eager — ConsolidationWorker::run_loop: poll + process on every interval
//!   lazy  — call run_once() at recall time (invoked from the CLI/MCP server)

pub mod extractive;
pub mod openai;

pub use extractive::ExtractiveConsolidator;
pub use openai::OpenAiConsolidator;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use store::Store;

// ─── output types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Foresight {
    pub statement: String,
    /// ISO-8601 date when this fact may no longer hold, or `None` if durable.
    pub expires: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ConsolidatedMemory {
    pub topic_path: String,
    pub title: String,
    pub body: String,
    pub foresight: Vec<Foresight>,
}

// ─── trait ───────────────────────────────────────────────────────────────────

/// LLM-free or LLM-backed consolidation model.
///
/// Manual `Pin<Box<dyn Future>>` form so the trait is object-safe without
/// requiring an external `async_trait` macro.
pub trait ConsolidationModel: Send + Sync {
    fn consolidate<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> Pin<Box<dyn Future<Output = Result<ConsolidatedMemory>> + Send + 'a>>;

    fn name(&self) -> &'static str;

    /// Max input+output tokens this model can handle in one `consolidate`
    /// call. The worker uses this to size clusters: it accumulates pending
    /// messages per-agent until the next one would exceed the budget, then
    /// flushes — so cluster size adapts to whatever model is plugged in
    /// instead of an arbitrary fixed message count.
    fn context_window(&self) -> usize;
}

// ─── worker ──────────────────────────────────────────────────────────────────

/// Background worker: polls for un-consolidated memories, runs the model,
/// writes results back via `Store::store_consolidation`.
pub struct ConsolidationWorker {
    pub store: Arc<Store>,
    pub model: Arc<dyn ConsolidationModel>,
    /// How many messages to process per tick.
    pub batch: i64,
    /// Seconds between ticks (eager mode).
    pub interval_secs: u64,
}

impl ConsolidationWorker {
    /// Process up to `self.batch` pending messages once: group them into
    /// per-agent clusters (never mixing two agents' memories into one call —
    /// that would break scope isolation), then run one `consolidate` call per
    /// cluster, fanned out across `CONCURRENCY` in-flight calls.
    ///
    /// Cluster boundary: fixed at `CLUSTER_SIZE` messages — matches
    /// EverMemOS's "Fixed-Message-10" ablation baseline, which scored within
    /// ~1-2 points of their full LLM-based semantic-boundary detector on both
    /// LoCoMo and LongMemEval, with no extra model call needed to find
    /// boundaries. The model's context window is only a safety ceiling here
    /// (`input_budget`), not the sizing target — EverMemOS's own ablation
    /// shows accuracy degrading already at fixed 1024-token chunks (Table 3:
    /// 84.52/75.19 vs 88.05/80.95 at Fixed-Message-10), so sizing clusters up
    /// to a 272K-token context window (as a first attempt here did) produces
    /// one lossy mega-summary per agent instead of real consolidation.
    ///
    /// Returns the number of messages *successfully* consolidated this tick
    /// (not the number attempted) — a failed cluster's messages never get a
    /// row, so they stay in `pending_consolidation` and are retried (as part
    /// of a new cluster) on the next `run_once` call. Callers that loop on
    /// this until "drained" must check for zero *successful* progress, not
    /// zero attempted, or a persistently failing message turns into an
    /// unbounded zero-backoff hot retry loop against the model API.
    pub async fn run_once(&self) -> Result<usize> {
        const CONCURRENCY: usize = 50;
        const CLUSTER_SIZE: usize = 10;
        // Reserve room for the system prompt and the model's JSON response —
        // the context window is shared between prompt and completion. Only
        // matters as a safety ceiling; CLUSTER_SIZE is the real boundary.
        const OUTPUT_RESERVE: usize = 8_000;
        const SYSTEM_PROMPT_RESERVE: usize = 500;
        const CHARS_PER_TOKEN: usize = 4;

        let pending = self
            .store
            .pending_consolidation(self.model.name(), self.batch)
            .await?;
        let input_budget = self
            .model
            .context_window()
            .saturating_sub(OUTPUT_RESERVE)
            .saturating_sub(SYSTEM_PROMPT_RESERVE);

        // `pending` is ordered by (agent_id, ts DESC), so each agent's rows
        // are contiguous — a single forward pass can build clusters.
        struct Cluster {
            ids: Vec<i64>,
            texts: Vec<String>,
        }
        let mut clusters: Vec<Cluster> = Vec::new();
        let mut cur: Option<(String, Cluster, usize)> = None; // (agent_id, cluster, tokens_used)
        for (id, agent_id, text) in pending {
            let cost = text.len() / CHARS_PER_TOKEN + 1;
            let starts_new = match &cur {
                Some((cur_agent, cluster, used)) => {
                    *cur_agent != agent_id
                        || cluster.ids.len() >= CLUSTER_SIZE
                        || *used + cost > input_budget
                }
                None => true,
            };
            if starts_new {
                if let Some((_, cluster, _)) = cur.take() {
                    clusters.push(cluster);
                }
                cur = Some((agent_id, Cluster { ids: vec![id], texts: vec![text] }, cost));
            } else if let Some((_, cluster, used)) = &mut cur {
                cluster.ids.push(id);
                cluster.texts.push(text);
                *used += cost;
            }
        }
        if let Some((_, cluster, _)) = cur.take() {
            clusters.push(cluster);
        }

        let sem = Arc::new(tokio::sync::Semaphore::new(CONCURRENCY));
        let mut js = tokio::task::JoinSet::new();
        for cluster in clusters {
            let sem = sem.clone();
            let model = self.model.clone();
            let store = self.store.clone();
            js.spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let refs: Vec<&str> = cluster.texts.iter().map(|s| s.as_str()).collect();
                match model.consolidate(&refs).await {
                    Ok(c) => {
                        let fj = serde_json::to_string(&c.foresight)
                            .unwrap_or_else(|_| "[]".to_string());
                        let mut succeeded = 0usize;
                        for id in &cluster.ids {
                            match store
                                .store_consolidation(*id, &c.topic_path, &c.title, &c.body, &fj, model.name())
                                .await
                            {
                                Ok(_) => succeeded += 1,
                                Err(e) => eprintln!("store_consolidation failed for id={id}: {e}"),
                            }
                        }
                        succeeded
                    }
                    Err(e) => {
                        eprintln!("consolidation failed for cluster (ids={:?}): {e}", cluster.ids);
                        0
                    }
                }
            });
        }
        let mut succeeded = 0usize;
        while let Some(res) = js.join_next().await {
            succeeded += res.unwrap_or(0);
        }

        Ok(succeeded)
    }

    /// Eager mode: run forever, ticking every `interval_secs`.
    pub async fn run_loop(&self) {
        loop {
            match self.run_once().await {
                Ok(0) => {} // nothing to do, just sleep
                Ok(n) => println!("[consolidation] processed {n} memories"),
                Err(e) => eprintln!("[consolidation] error: {e}"),
            }
            tokio::time::sleep(Duration::from_secs(self.interval_secs)).await;
        }
    }
}

// ─── mode ────────────────────────────────────────────────────────────────────

/// The three consolidation strategies for the three-way eval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsolidationMode {
    /// No consolidation — use raw stored text (baseline).
    None,
    /// Consolidate at recall time on each retrieved hit.
    Lazy,
    /// Consolidation pre-ran by the worker; use stored consolidated body.
    Eager,
}

impl std::str::FromStr for ConsolidationMode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "none" => Ok(Self::None),
            "lazy" => Ok(Self::Lazy),
            "eager" => Ok(Self::Eager),
            _ => anyhow::bail!("unknown mode '{s}': use none, lazy, or eager"),
        }
    }
}

// ─── timestamp helper (shared) ───────────────────────────────────────────────

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}
