//! M5 "big test": LongMemEval (ICLR 2025, arXiv:2410.10813) three-way
//! consolidation eval — none / lazy / eager — on top of the same
//! retrieval pipeline exercised by `locomo_qa`.
//!
//! Unlike LoCoMo (one agent, many questions), LongMemEval is one agent
//! ("haystack") per question: 500 independent instances, each with ~40-60
//! sessions (~490 turns) to search through for a single question. Gold is
//! at session granularity (`answer_session_ids`), not per-turn.
//!
//! Modes:
//!   none  — raw retrieved turn text is the QA context (baseline).
//!   lazy  — the top-k retrieved candidates are consolidated into ONE
//!           synthesized memory at query time (1 LLM call/question).
//!   eager — the ENTIRE haystack is pre-consolidated per-message by a
//!           `ConsolidationWorker` before any question is asked (1 LLM
//!           call/message); QA context substitutes each hit's stored
//!           title+body for its raw text.
//!
//! Data: download `longmemeval_s_cleaned.json` (small variant, ~40-60
//! sessions/instance) from HuggingFace into data/longmemeval_s.json:
//!   curl -L -o data/longmemeval_s.json \
//!     https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
//!
//! Embedding: requires Google (Vertex AI service account or GOOGLE_TOKEN) —
//! the haystacks are too large for serial local fastembed. Set
//! GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json or GOOGLE_TOKEN=...
//!
//! Run:
//!   GOOGLE_APPLICATION_CREDENTIALS=... cargo run --release --bin longmemeval_qa -- --n 50 --k 20
//!   GOOGLE_APPLICATION_CREDENTIALS=... OPENAI_API_KEY=sk-... \
//!     cargo run --release --bin longmemeval_qa -- --n 50 --k 20 --mode eager --qa

use anyhow::Result;
use consolidation::{ConsolidationModel, ConsolidationWorker, ExtractiveConsolidator, OpenAiConsolidator};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::str::FromStr;
use std::sync::Arc;
use store::google_embed::{cache_load, cache_save, google_auth_from_env, google_embed_batch};
use store::{rrf, Config, Store};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    None,
    Lazy,
    Eager,
}

impl FromStr for Mode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "none" => Ok(Self::None),
            "lazy" => Ok(Self::Lazy),
            "eager" => Ok(Self::Eager),
            other => anyhow::bail!("unknown mode '{other}': use none, lazy, or eager"),
        }
    }
}

struct QaItem {
    agent: String,
    question: String,
    answer: String,
    qtype: String,
    gold_ids: HashSet<i64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut n = 50usize;
    let mut k = 20usize;
    let mut mode = Mode::None;
    let mut cmodel_name = "openai".to_string();
    let mut qa_mode = false;
    let mut max_q = usize::MAX;
    let mut resume = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--n" => n = args.next().and_then(|s| s.parse().ok()).unwrap_or(n),
            "--k" => k = args.next().and_then(|s| s.parse().ok()).unwrap_or(k),
            "--mode" => mode = Mode::from_str(&args.next().unwrap_or_default())?,
            "--consolidation-model" => cmodel_name = args.next().unwrap_or(cmodel_name),
            "--qa" => qa_mode = true,
            "--max-q" => max_q = args.next().and_then(|s| s.parse().ok()).unwrap_or(max_q),
            "--resume" => resume = true,
            _ => {}
        }
    }

    let store = Arc::new(Store::connect(Config::load("memoryd.toml")?).await?);

    let client = reqwest::Client::new();
    let (google_key, google_project): (Option<String>, Option<String>) =
        match google_auth_from_env(&client).await? {
            Some((t, p)) => (Some(t), p),
            None => (None, None),
        };
    let google_key = google_key.ok_or_else(|| {
        anyhow::anyhow!(
            "LongMemEval haystacks are too large for serial local fastembed. \
             Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_TOKEN."
        )
    })?;
    let openai_key = std::env::var("OPENAI_API_KEY").ok().filter(|s| !s.is_empty());
    if qa_mode && openai_key.is_none() {
        anyhow::bail!("--qa requires OPENAI_API_KEY");
    }
    if mode != Mode::None && cmodel_name == "openai" && openai_key.is_none() {
        anyhow::bail!("--mode lazy/eager with --consolidation-model openai requires OPENAI_API_KEY");
    }

    println!("loading data/longmemeval_s.json (--n {n} instances, stratified)...");
    let raw = std::fs::read_to_string("data/longmemeval_s.json")
        .map_err(|e| anyhow::anyhow!("{e}: run the curl command in this file's header comment first"))?;
    let data: Value = serde_json::from_str(&raw)?;
    let samples = data.as_array().expect("top level must be an array");

    // Stratified sample: proportional-to-population quota per question_type,
    // deterministic (first-in-file-order within each type), so repeated runs
    // with the same --n select the identical subset across mode comparisons.
    // The raw file is NOT shuffled (long runs of one type in a row), so a
    // plain .take(n) would badly bias small samples toward one category.
    let selected_indices: Vec<usize> = {
        let mut by_type: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, s) in samples.iter().enumerate() {
            let qtype = s["question_type"].as_str().unwrap_or("").to_string();
            by_type.entry(qtype).or_default().push(i);
        }
        let mut types: Vec<&String> = by_type.keys().collect();
        types.sort(); // stable category order across runs
        let total = samples.len();
        let mut quotas: Vec<(String, usize)> = types
            .iter()
            .map(|t| {
                let pop = by_type[*t].len();
                ((*t).clone(), ((n * pop) as f64 / total as f64).round() as usize)
            })
            .collect();
        // Largest-remainder-free fixup: nudge the largest category to absorb
        // any rounding drift so quotas sum to exactly n.
        let drift = n as i64 - quotas.iter().map(|(_, q)| *q as i64).sum::<i64>();
        if let Some(biggest) = quotas.iter_mut().max_by_key(|(t, _)| by_type[t].len()) {
            biggest.1 = (biggest.1 as i64 + drift).max(0) as usize;
        }
        let mut idx: Vec<usize> = quotas
            .into_iter()
            .flat_map(|(t, q)| by_type[&t].iter().take(q).copied().collect::<Vec<_>>())
            .collect();
        idx.sort_unstable();
        idx
    };

    // ---- Phase 1: parse instances, flatten turns. ----
    struct TurnRec {
        agent: String,
        text: String,
    }
    struct Pending {
        agent: String,
        question: String,
        answer: String,
        qtype: String,
        answer_session_ids: HashSet<String>,
    }

    let mut turn_recs: Vec<TurnRec> = Vec::new();
    // agent -> [(session_id, index into turn_recs)]
    let mut agent_session_idx: HashMap<String, Vec<(String, usize)>> = HashMap::new();
    let mut pending: Vec<Pending> = Vec::new();

    for &si in &selected_indices {
        let s = &samples[si];
        let qid = s["question_id"].as_str().unwrap_or("q").to_string();
        let agent = format!("lme_{qid}");
        let qtype = s["question_type"].as_str().unwrap_or("").to_string();
        let question = s["question"].as_str().unwrap_or("").to_string();
        let answer = match &s["answer"] {
            Value::String(x) => x.clone(),
            other => other.to_string(),
        };
        let answer_session_ids: HashSet<String> = s["answer_session_ids"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(|x| x.to_string())).collect())
            .unwrap_or_default();

        let session_ids = s["haystack_session_ids"].as_array().cloned().unwrap_or_default();
        let dates = s["haystack_dates"].as_array().cloned().unwrap_or_default();
        let sessions = s["haystack_sessions"].as_array().cloned().unwrap_or_default();

        for (i, sess) in sessions.iter().enumerate() {
            let sess_id = session_ids.get(i).and_then(|v| v.as_str()).unwrap_or("").to_string();
            let date = dates.get(i).and_then(|v| v.as_str()).unwrap_or("");
            if let Some(turns) = sess.as_array() {
                for t in turns {
                    let role = t["role"].as_str().unwrap_or("user");
                    let content = t["content"].as_str().unwrap_or("");
                    if content.is_empty() {
                        continue;
                    }
                    let text = format!("[{date}] {role}: {content}");
                    let idx = turn_recs.len();
                    turn_recs.push(TurnRec { agent: agent.clone(), text });
                    agent_session_idx.entry(agent.clone()).or_default().push((sess_id.clone(), idx));
                }
            }
        }

        pending.push(Pending { agent, question, answer, qtype, answer_session_ids });
    }
    println!(
        "{} instances, {} total turns ({:.0} turns/instance avg)",
        pending.len(),
        turn_recs.len(),
        turn_recs.len() as f64 / pending.len() as f64
    );

    // ---- Phase 2: batch-embed turns (Google, disk-cached). ----
    let turn_texts: Vec<String> = turn_recs.iter().map(|t| t.text.clone()).collect();
    let turn_cache_path = format!("data/cache_lme_turns_n{n}_{}t.bin", turn_texts.len());
    let turn_embs: Vec<Vec<f32>> = if let Some(cached) = cache_load(&turn_cache_path, turn_texts.len()) {
        println!("loaded {} turn embeddings from cache ({turn_cache_path})", cached.len());
        cached
    } else {
        println!("embedding {} turns via Google...", turn_texts.len());
        let embs = google_embed_batch(&client, &google_key, google_project.as_deref(), &turn_texts, "RETRIEVAL_DOCUMENT").await?;
        cache_save(&turn_cache_path, &embs)?;
        println!("  saved to {turn_cache_path}");
        embs
    };

    // ---- Phase 3: ingest (raw, no gate — this loads the episodic substrate). ----
    let dim = turn_embs.first().map(|v| v.len()).unwrap_or(384);
    let message_ids: Vec<i64> = if resume {
        println!("--resume: skipping reset + re-ingest, recovering message ids from existing DB rows...");
        let mut ids = Vec::with_capacity(turn_recs.len());
        let mut i = 0;
        while i < turn_recs.len() {
            let agent = &turn_recs[i].agent;
            let mut j = i;
            while j < turn_recs.len() && turn_recs[j].agent == *agent {
                j += 1;
            }
            let expected = j - i;
            let rows = store.messages_for_agent(agent).await?;
            if rows.len() != expected {
                anyhow::bail!(
                    "--resume mismatch for agent {agent}: expected {expected} messages, found {} \
                     in DB. The data/--n must match the interrupted run exactly; otherwise drop \
                     --resume to start fresh.",
                    rows.len()
                );
            }
            ids.extend(rows.into_iter().map(|(id, _)| id));
            i = j;
        }
        println!("  recovered {} message ids across {} agents", ids.len(), pending.len());
        ids
    } else {
        println!("resetting schema for {dim}-dim embeddings...");
        store.reset_for_dim(dim).await?;
        println!("inserting {} turns into DB...", turn_recs.len());
        let mut ids: Vec<i64> = Vec::with_capacity(turn_recs.len());
        for (i, (tr, emb)) in turn_recs.iter().zip(turn_embs.into_iter()).enumerate() {
            if i % 2000 == 0 {
                print!("\r  [{i}/{}]", turn_recs.len());
                let _ = std::io::stdout().flush();
            }
            let id = store.store_raw_vec(&tr.agent, "private", "user", &tr.text, emb).await?;
            ids.push(id);
        }
        println!("\r  [{0}/{0}]", turn_recs.len());
        ids
    };

    // agent -> session_id -> [message_id]
    let mut session_msg_ids: HashMap<String, HashMap<String, Vec<i64>>> = HashMap::new();
    for (agent, list) in &agent_session_idx {
        for (sess_id, idx) in list {
            session_msg_ids
                .entry(agent.clone())
                .or_default()
                .entry(sess_id.clone())
                .or_default()
                .push(message_ids[*idx]);
        }
    }

    // ---- Phase 4: resolve gold ids per instance from answer_session_ids. ----
    let mut qa_items: Vec<QaItem> = Vec::new();
    for p in pending {
        let gold_ids: HashSet<i64> = session_msg_ids
            .get(&p.agent)
            .map(|sess_map| {
                p.answer_session_ids
                    .iter()
                    .filter_map(|sid| sess_map.get(sid))
                    .flatten()
                    .copied()
                    .collect()
            })
            .unwrap_or_default();
        if gold_ids.is_empty() {
            eprintln!("warning: no gold ids resolved for {} — skipping", p.agent);
            continue;
        }
        qa_items.push(QaItem {
            agent: p.agent,
            question: p.question,
            answer: p.answer,
            qtype: p.qtype,
            gold_ids,
        });
    }
    println!("{} QA items with resolved gold ids", qa_items.len());

    // ---- Phase 5: optional EAGER consolidation pre-run (whole haystack). ----
    let cmodel: Option<Arc<dyn ConsolidationModel>> = if mode != Mode::None {
        Some(match cmodel_name.as_str() {
            "openai" => Arc::new(OpenAiConsolidator::new(openai_key.clone().unwrap())) as Arc<dyn ConsolidationModel>,
            "extractive" => Arc::new(ExtractiveConsolidator) as Arc<dyn ConsolidationModel>,
            other => anyhow::bail!("unknown consolidation model '{other}': use extractive or openai"),
        })
    } else {
        None
    };

    if mode == Mode::Eager {
        let cm = cmodel.clone().unwrap();
        println!("running eager consolidation (model={}) over {} messages...", cm.name(), turn_recs.len());
        let worker = ConsolidationWorker {
            store: store.clone(),
            model: cm.clone(),
            // Pull the entire pending pool in one sweep so per-agent
            // clusters never split across an arbitrary batch boundary.
            batch: turn_recs.len() as i64 + 1,
            interval_secs: 0,
        };
        let mut total = 0usize;
        let mut stale_rounds = 0u32;
        // Exponential backoff (2,4,8,16,32,64s, ~126s total budget) so a brief
        // rate-limit burst across many messages gets time to clear, rather
        // than the give-up path mistaking it for one permanently-broken
        // message and abandoning a whole batch after a few seconds.
        const MAX_STALE_ROUNDS: u32 = 6;
        loop {
            let remaining = store.pending_consolidation_count(cm.name()).await?;
            if remaining == 0 {
                break;
            }
            let done = worker.run_once().await?;
            total += done;
            print!("\r  consolidated {total}/{} ({remaining} pending)...   ", turn_recs.len());
            let _ = std::io::stdout().flush();
            if done == 0 {
                stale_rounds += 1;
                if stale_rounds >= MAX_STALE_ROUNDS {
                    println!(
                        "\n  no progress for {MAX_STALE_ROUNDS} backoff rounds — giving up with \
                         {remaining} messages permanently unconsolidated (eager QA context falls \
                         back to raw text for those)."
                    );
                    break;
                }
                let backoff = 2u64.pow(stale_rounds);
                tokio::time::sleep(std::time::Duration::from_secs(backoff)).await;
            } else {
                stale_rounds = 0;
            }
        }
        println!();
    }

    // ---- Phase 6: batch-embed questions (Google, disk-cached). ----
    let q_texts: Vec<String> = qa_items.iter().map(|q| q.question.clone()).collect();
    let q_cache_path = format!("data/cache_lme_questions_n{n}_{}q.bin", q_texts.len());
    let q_embs: Vec<Vec<f32>> = if let Some(cached) = cache_load(&q_cache_path, q_texts.len()) {
        println!("loaded {} question embeddings from cache", cached.len());
        cached
    } else {
        println!("embedding {} questions via Google...", q_texts.len());
        let embs = google_embed_batch(&client, &google_key, google_project.as_deref(), &q_texts, "RETRIEVAL_QUERY").await?;
        cache_save(&q_cache_path, &embs)?;
        embs
    };

    // ---- Phase 7: retrieval scoring (parallel — rerank is CPU-bound, so it
    // runs via spawn_blocking off the async worker threads; bounded to core
    // count so concurrent reranks don't oversubscribe the machine). This
    // used to be a plain sequential `for` loop, one rerank call at a time,
    // which on long LongMemEval documents made scoring the slowest phase of
    // the whole eval by a wide margin. ----
    let methods = ["bm25", "vector", "fused", "full"];
    let mut recall_sum: HashMap<(String, usize), f64> = HashMap::new();
    let mut count: HashMap<String, usize> = HashMap::new();

    struct QaTask {
        question: String,
        gold_answer: String,
        qtype: String,
        hits: Vec<store::Hit>,
    }
    struct ScoreResult {
        qtype: String,
        recalls: [f64; 4],
        qa_task: Option<QaTask>,
    }

    let total_questions = qa_items.len().min(max_q);
    let rerank_sem = Arc::new(tokio::sync::Semaphore::new(num_cpus()));
    let mut js = tokio::task::JoinSet::new();
    for (qi, q) in qa_items.iter().take(max_q).enumerate() {
        let store = store.clone();
        let sem = rerank_sem.clone();
        let emb = q_embs[qi].clone();
        let question = q.question.clone();
        let gold_answer = q.answer.clone();
        let qtype = q.qtype.clone();
        let agent = q.agent.clone();
        let gold_ids = q.gold_ids.clone();
        js.spawn(async move {
            let bm = store.bm25_search(&agent, &question, 100).await?;
            let vec = store.vector_search(&agent, &emb, 100).await?;
            let mut text: HashMap<i64, String> = HashMap::new();
            for (id, t) in bm.iter().chain(vec.iter()) {
                text.entry(*id).or_insert_with(|| t.clone());
            }
            let bm_ids: Vec<i64> = bm.iter().map(|(id, _)| *id).collect();
            let vec_ids: Vec<i64> = vec.iter().map(|(id, _)| *id).collect();
            let fused = rrf(&[bm_ids.clone(), vec_ids.clone()]);
            let cand_size = (k * 2).max(100).min(fused.len());
            let cand: Vec<(i64, String)> = fused.iter().take(cand_size).map(|id| (*id, text[id].clone())).collect();

            let _permit = sem.acquire().await.unwrap();
            let q_for_rerank = question.clone();
            let store2 = store.clone();
            let full = tokio::task::spawn_blocking(move || store2.rerank(&q_for_rerank, &cand, k)).await??;
            drop(_permit);

            let recall = |ids: &[i64]| -> f64 {
                let got = ids.iter().take(k).filter(|i| gold_ids.contains(i)).count();
                got as f64 / gold_ids.len() as f64
            };
            let full_ids: Vec<i64> = full.iter().map(|h| h.id).collect();
            let recalls = [recall(&bm_ids), recall(&vec_ids), recall(&fused), recall(&full_ids)];

            Ok::<ScoreResult, anyhow::Error>(ScoreResult {
                qtype: qtype.clone(),
                recalls,
                qa_task: if qa_mode {
                    Some(QaTask { question, gold_answer, qtype, hits: full })
                } else {
                    None
                },
            })
        });
    }

    let mut qa_tasks: Vec<QaTask> = Vec::new();
    let mut finished = 0usize;
    while let Some(res) = js.join_next().await {
        finished += 1;
        if finished == 1 || finished % 10 == 0 || finished == total_questions {
            print!("\r  [{finished}/{total_questions}] scoring...");
            let _ = std::io::stdout().flush();
        }
        let r = res??;
        for (mi, rec) in r.recalls.into_iter().enumerate() {
            *recall_sum.entry((r.qtype.clone(), mi)).or_default() += rec;
            *recall_sum.entry(("ALL".to_string(), mi)).or_default() += rec;
        }
        *count.entry(r.qtype.clone()).or_default() += 1;
        *count.entry("ALL".to_string()).or_default() += 1;
        if let Some(t) = r.qa_task {
            qa_tasks.push(t);
        }
    }
    println!();

    // ---- Phase 8: QA evaluation (parallel — IO-bound LLM calls). ----
    let mut qa_correct: HashMap<String, usize> = HashMap::new();
    if qa_mode && !qa_tasks.is_empty() {
        println!("running {} QA items (mode={mode:?}, 30 concurrent)...", qa_tasks.len());
        let sem = Arc::new(tokio::sync::Semaphore::new(30));
        let client = Arc::new(client);
        let key = Arc::new(openai_key.clone().unwrap());
        let store = store.clone();
        let cmodel = cmodel.clone();
        let cmodel_name = Arc::new(cmodel_name.clone());
        let mut js = tokio::task::JoinSet::new();
        for task in qa_tasks {
            let sem = sem.clone();
            let c = client.clone();
            let k = key.clone();
            let store = store.clone();
            let cmodel = cmodel.clone();
            let cmodel_name = cmodel_name.clone();
            js.spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let context = build_context(mode, &store, cmodel.as_deref(), &cmodel_name, &task.hits).await?;
                let pred = answer(&c, &k, &task.question, &context).await?;
                let ok = judge(&c, &k, &task.question, &task.gold_answer, &pred).await?;
                Ok::<(String, bool), anyhow::Error>((task.qtype, ok))
            });
        }
        let mut finished = 0usize;
        while let Some(res) = js.join_next().await {
            finished += 1;
            if finished % 25 == 0 || finished == total_questions {
                print!("\r  QA [{finished}/{total_questions}]...");
                let _ = std::io::stdout().flush();
            }
            let (qtype, ok) = res??;
            if ok {
                *qa_correct.entry(qtype).or_default() += 1;
                *qa_correct.entry("ALL".to_string()).or_default() += 1;
            }
        }
        println!();
    }

    // ---- Report. ----
    let cat_order = [
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
        "ALL",
    ];
    println!("\n=== LongMemEval retrieval recall@{k} (Google text-embedding-004, {dim}-dim) ===");
    println!("{:<28} {:>8} {:>8} {:>8} {:>8}  n", "category", "bm25", "vector", "fused", "full");
    for c in cat_order {
        let n = *count.get(c).unwrap_or(&0);
        if n == 0 {
            continue;
        }
        print!("{c:<28}");
        for mi in 0..methods.len() {
            let v = recall_sum.get(&(c.to_string(), mi)).copied().unwrap_or(0.0) / n as f64;
            print!(" {:>7.1}%", v * 100.0);
        }
        println!("  {n}");
    }

    if qa_mode {
        println!("\n=== QA accuracy (gpt-5-mini answer + judge, mode={mode:?}, consolidation-model={cmodel_name}) ===");
        println!("{:<28} {:>8}  n", "category", "accuracy");
        for c in cat_order {
            let n = *count.get(c).unwrap_or(&0);
            if n == 0 {
                continue;
            }
            let acc = *qa_correct.get(c).unwrap_or(&0) as f64 / n as f64 * 100.0;
            println!("{c:<28} {:>7.1}%  {n}", acc);
        }
    } else {
        println!("\n({total_questions} questions scored for retrieval. Re-run with --qa + OPENAI_API_KEY for QA accuracy.)");
    }
    Ok(())
}

/// Build the QA reader's context string for one question's top-k hits,
/// per the active consolidation mode.
async fn build_context(
    mode: Mode,
    store: &Store,
    cmodel: Option<&dyn ConsolidationModel>,
    cmodel_name: &str,
    hits: &[store::Hit],
) -> Result<String> {
    match mode {
        Mode::None => Ok(hits.iter().map(|h| h.text.clone()).collect::<Vec<_>>().join("\n")),
        Mode::Eager => {
            // Multiple hits can land in the same cluster and share an
            // identical consolidation row — include each distinct one once,
            // not duplicated per hit.
            let mut seen = HashSet::new();
            let mut parts = Vec::with_capacity(hits.len());
            for h in hits {
                match store.get_consolidation(h.id, cmodel_name).await? {
                    Some((_, title, body, _)) => {
                        if seen.insert(body.clone()) {
                            parts.push(format!("{title}: {body}"));
                        }
                    }
                    None => parts.push(h.text.clone()),
                }
            }
            Ok(parts.join("\n"))
        }
        Mode::Lazy => {
            let texts: Vec<&str> = hits.iter().map(|h| h.text.as_str()).collect();
            let c = cmodel.expect("lazy mode requires a consolidation model").consolidate(&texts).await?;
            let mut out = format!("{}: {}", c.title, c.body);
            for f in &c.foresight {
                out.push_str(&format!("\n- (foresight) {}", f.statement));
            }
            Ok(out)
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

async fn chat(client: &reqwest::Client, key: &str, system: &str, user: &str) -> Result<String> {
    let req = json!({
        "model": "gpt-5-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    });
    let resp = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(key)
        .json(&req)
        .send()
        .await?;
    let body: Value = resp.json().await?;
    Ok(body["choices"][0]["message"]["content"].as_str().unwrap_or("").trim().to_string())
}

async fn answer(client: &reqwest::Client, key: &str, question: &str, context: &str) -> Result<String> {
    chat(
        client,
        key,
        "Answer the question using ONLY the provided memories. Be concise: a few words or a short phrase. For dates, use the format shown in the memories.",
        &format!("Memories:\n{context}\n\nQuestion: {question}\nAnswer:"),
    )
    .await
}

async fn judge(client: &reqwest::Client, key: &str, question: &str, gold: &str, pred: &str) -> Result<bool> {
    let v = chat(
        client,
        key,
        "You are grading a model answer against a reference answer. Reply with exactly one word: CORRECT or WRONG.",
        &format!("Question: {question}\nReference answer: {gold}\nModel answer: {pred}\nIs the model answer correct (same meaning as the reference)?"),
    )
    .await?;
    Ok(v.to_uppercase().contains("CORRECT"))
}
