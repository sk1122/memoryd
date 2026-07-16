//! M3 "big test": the LoCoMo retrieval + QA benchmark, end-to-end.
//!
//! Two parts:
//!   A. Retrieval recall@k (NO LLM) — for each question, does the pipeline
//!      surface the gold evidence turns? Reported as an ablation:
//!      BM25-only / vector-only / RRF-fused / RRF+rerank. This runs offline.
//!   B. QA accuracy (needs OPENAI_API_KEY, --qa flag) — answer each question
//!      from the retrieved context with gpt-5-mini, judge against gold.
//!      Mirrors TrueMemory's eval (categories 1-4, adversarial excluded).
//!
//! Embedding: set GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json to use
//! Gemini text-embedding-004 (fast, batched) via service account auth.
//! Or set GOOGLE_TOKEN directly with a bearer token.
//! Otherwise falls back to local fastembed (slow, one-at-a-time).
//!
//! Run:  cargo run -p store --release --bin locomo_qa -- --convs 10 --k 20
//!       GOOGLE_APPLICATION_CREDENTIALS=/path/key.json cargo run -p store --release --bin locomo_qa -- --convs 10 --k 20
//!       OPENAI_API_KEY=sk-... cargo run -p store --release --bin locomo_qa -- --qa

use anyhow::Result;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::Arc;
use store::google_embed::{cache_load, cache_save, google_auth_from_env, google_embed_batch};
use store::{rrf, Config, Store};

struct Qa {
    question: String,
    answer: String,
    category: i64,
    gold_ids: HashSet<i64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut convs = 10usize;
    let mut k = 20usize;
    let mut qa_mode = false;
    let mut max_q = usize::MAX;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--convs" => convs = args.next().and_then(|s| s.parse().ok()).unwrap_or(convs),
            "--k" => k = args.next().and_then(|s| s.parse().ok()).unwrap_or(k),
            "--max-q" => max_q = args.next().and_then(|s| s.parse().ok()).unwrap_or(max_q),
            "--qa" => qa_mode = true,
            _ => {}
        }
    }

    let store = Store::connect(Config::load("memoryd.toml")?).await?;

    let data: Value = serde_json::from_str(&std::fs::read_to_string("data/locomo10.json")?)?;
    let samples = data.as_array().unwrap();

    let client = reqwest::Client::new();
    // (token, project_id) — project_id is Some only for service account / Vertex AI
    let (google_key, google_project): (Option<String>, Option<String>) =
        match google_auth_from_env(&client).await? {
            Some((t, p)) => (Some(t), p),
            None => (None, None),
        };
    let openai_key = std::env::var("OPENAI_API_KEY").ok().filter(|s| !s.is_empty());

    if qa_mode && openai_key.is_none() {
        anyhow::bail!("--qa requires OPENAI_API_KEY");
    }

    // ---- Phase 1: Parse all conversations (no DB yet). ----
    // Collect turns and QAs without embedding so we can batch-embed later.
    struct TurnRec {
        agent: String,
        dia_id: String,
        text: String,
    }
    struct QaRec {
        agent: String,
        question: String,
        answer: String,
        category: i64,
        evidence_dia_ids: Vec<String>,
    }

    let mut turn_recs: Vec<TurnRec> = Vec::new();
    let mut qa_recs: Vec<QaRec> = Vec::new();
    let mut agent_order: Vec<String> = Vec::new();

    for s in samples.iter().take(convs) {
        let agent = s["sample_id"].as_str().unwrap_or("conv").to_string();
        agent_order.push(agent.clone());
        let conv = &s["conversation"];
        let obj = conv.as_object().unwrap();
        let mut nums: Vec<u64> = obj
            .keys()
            .filter_map(|kk| kk.strip_prefix("session_").and_then(|r| r.parse().ok()))
            .collect();
        nums.sort_unstable();

        for n in nums {
            let date = conv[format!("session_{n}_date_time")].as_str().unwrap_or("");
            if let Some(turns) = conv[format!("session_{n}")].as_array() {
                for t in turns {
                    let raw = t["text"].as_str().unwrap_or("");
                    if raw.is_empty() {
                        continue;
                    }
                    let speaker = t["speaker"].as_str().unwrap_or("");
                    let dia_id = t["dia_id"].as_str().unwrap_or("").to_string();
                    let text = format!("[{date}] {speaker}: {raw}");
                    turn_recs.push(TurnRec { agent: agent.clone(), dia_id, text });
                }
            }
        }

        for q in s["qa"].as_array().unwrap_or(&vec![]) {
            let category = q["category"].as_i64().unwrap_or(0);
            if category == 5 {
                continue;
            }
            let evidence_dia_ids: Vec<String> = q["evidence"]
                .as_array()
                .map(|ev| {
                    ev.iter()
                        .filter_map(|e| e.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            if evidence_dia_ids.is_empty() {
                continue;
            }
            qa_recs.push(QaRec {
                agent: agent.clone(),
                question: q["question"].as_str().unwrap_or("").to_string(),
                answer: match &q["answer"] {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                },
                category,
                evidence_dia_ids,
            });
        }
    }

    // ---- Phase 2: Batch-embed all turn texts (with disk cache). ----
    let turn_texts: Vec<String> = turn_recs.iter().map(|t| t.text.clone()).collect();
    let turn_cache_path = format!("data/cache_turns_{}turns.bin", turn_texts.len());
    let turn_embs: Vec<Vec<f32>> = if let Some(cached) = cache_load(&turn_cache_path, turn_texts.len()) {
        println!("loaded {} turn embeddings from cache ({})", cached.len(), turn_cache_path);
        cached
    } else {
        println!("embedding {} turns...", turn_texts.len());
        let embs = if let Some(key) = &google_key {
            google_embed_batch(&client, key, google_project.as_deref(), &turn_texts, "RETRIEVAL_DOCUMENT").await?
        } else {
            println!("  (no GOOGLE_TOKEN, using local fastembed — this is slow)");
            let mut v = Vec::with_capacity(turn_texts.len());
            for (i, t) in turn_texts.iter().enumerate() {
                if i % 100 == 0 { print!("\r  [{i}/{}]", turn_texts.len()); let _ = std::io::stdout().flush(); }
                v.push(store.embed(t)?);
            }
            println!();
            v
        };
        cache_save(&turn_cache_path, &embs)?;
        println!("  saved to {turn_cache_path}");
        embs
    };

    // ---- Phase 3: Ingest with pre-computed embeddings. ----
    let dim = turn_embs.first().map(|v| v.len()).unwrap_or(384);
    println!("resetting schema for {dim}-dim embeddings...");
    store.reset_for_dim(dim).await?;
    println!("inserting turns into DB...");
    let mut dia_maps: HashMap<String, HashMap<String, i64>> = HashMap::new();
    for (tr, emb) in turn_recs.iter().zip(turn_embs.into_iter()) {
        let id = store.store_raw_vec(&tr.agent, "private", "user", &tr.text, emb).await?;
        dia_maps
            .entry(tr.agent.clone())
            .or_default()
            .insert(tr.dia_id.clone(), id);
    }

    // ---- Phase 4: Build Qa structs, resolve gold_ids. ----
    let mut convs_qa: Vec<(String, Vec<Qa>)> = Vec::new();
    {
        let mut by_agent: HashMap<String, Vec<Qa>> = HashMap::new();
        for qr in &qa_recs {
            let dia_map = dia_maps.get(&qr.agent);
            let gold_ids: HashSet<i64> = qr
                .evidence_dia_ids
                .iter()
                .filter_map(|d| dia_map.and_then(|m| m.get(d)).copied())
                .collect();
            if gold_ids.is_empty() {
                continue;
            }
            by_agent.entry(qr.agent.clone()).or_default().push(Qa {
                question: qr.question.clone(),
                answer: qr.answer.clone(),
                category: qr.category,
                gold_ids,
            });
        }
        for agent in &agent_order {
            if let Some(qas) = by_agent.remove(agent) {
                let n_turns = dia_maps.get(agent).map(|m| m.len()).unwrap_or(0);
                println!("ingested {agent}: {n_turns} turns, {} questions", qas.len());
                convs_qa.push((agent.clone(), qas));
            }
        }
    }

    // ---- Phase 5: Batch-embed all questions (with disk cache). ----
    let all_q_texts: Vec<String> = convs_qa
        .iter()
        .flat_map(|(_, qas)| qas.iter().map(|q| q.question.clone()))
        .collect();
    let q_cache_path = format!("data/cache_questions_{}questions.bin", all_q_texts.len());
    let all_q_embs: Vec<Vec<f32>> = if let Some(cached) = cache_load(&q_cache_path, all_q_texts.len()) {
        println!("loaded {} question embeddings from cache ({})", cached.len(), q_cache_path);
        cached
    } else {
        println!("embedding {} questions...", all_q_texts.len());
        let embs = if let Some(key) = &google_key {
            google_embed_batch(&client, key, google_project.as_deref(), &all_q_texts, "RETRIEVAL_QUERY").await?
        } else {
            let mut v = Vec::with_capacity(all_q_texts.len());
            for (i, q) in all_q_texts.iter().enumerate() {
                if i % 100 == 0 { print!("\r  [{i}/{}]", all_q_texts.len()); let _ = std::io::stdout().flush(); }
                v.push(store.embed(q)?);
            }
            println!();
            v
        };
        cache_save(&q_cache_path, &embs)?;
        println!("  saved to {q_cache_path}");
        embs
    };

    // ---- Phase 6a: Retrieval scoring (sequential — reranker is CPU-bound). ----
    let methods = ["bm25", "vector", "fused", "full"];
    let mut recall_sum: HashMap<(i64, usize), f64> = HashMap::new();
    let mut count: HashMap<i64, usize> = HashMap::new();
    let mut total_q = 0usize;
    let total_questions: usize = convs_qa.iter().map(|(_, qas)| qas.len()).sum::<usize>().min(max_q);
    let mut done = 0usize;
    let mut q_emb_offset = 0usize;

    // Collected for parallel QA pass below.
    struct QaTask { question: String, gold_answer: String, context: String, category: i64 }
    let mut qa_tasks: Vec<QaTask> = Vec::new();

    'outer: for (agent, qas) in &convs_qa {
        for (qi, q) in qas.iter().enumerate() {
            if done >= max_q { break 'outer; }
            done += 1;
            if done == 1 || done % 50 == 0 || done == total_questions {
                print!("\r  [{done}/{total_questions}] scoring...");
                let _ = std::io::stdout().flush();
            }

            let emb = &all_q_embs[q_emb_offset + qi];
            let bm = store.bm25_search(agent, &q.question, 100).await?;
            let vec = store.vector_search(agent, emb, 100).await?;
            let mut text: HashMap<i64, String> = HashMap::new();
            for (id, t) in bm.iter().chain(vec.iter()) {
                text.entry(*id).or_insert_with(|| t.clone());
            }
            let bm_ids: Vec<i64> = bm.iter().map(|(id, _)| *id).collect();
            let vec_ids: Vec<i64> = vec.iter().map(|(id, _)| *id).collect();
            let fused = rrf(&[bm_ids.clone(), vec_ids.clone()]);
            // Feed at least 2× k candidates so reranker can always return k results.
            let cand_size = (k * 2).max(100).min(fused.len());
            let cand: Vec<(i64, String)> = fused
                .iter()
                .take(cand_size)
                .map(|id| (*id, text[id].clone()))
                .collect();
            let full = store.rerank(&q.question, &cand, k)?;

            let recall = |ids: &[i64]| -> f64 {
                let got = ids.iter().take(k).filter(|i| q.gold_ids.contains(i)).count();
                got as f64 / q.gold_ids.len() as f64
            };
            let full_ids: Vec<i64> = full.iter().map(|h| h.id).collect();
            for (mi, r) in [
                recall(&bm_ids),
                recall(&vec_ids),
                recall(&fused),
                recall(&full_ids),
            ]
            .into_iter()
            .enumerate()
            {
                *recall_sum.entry((q.category, mi)).or_default() += r;
                *recall_sum.entry((0, mi)).or_default() += r;
            }
            *count.entry(q.category).or_default() += 1;
            *count.entry(0).or_default() += 1;
            total_q += 1;

            if qa_mode {
                let context = full.iter().map(|h| h.text.clone()).collect::<Vec<_>>().join("\n");
                qa_tasks.push(QaTask {
                    question: q.question.clone(),
                    gold_answer: q.answer.clone(),
                    context,
                    category: q.category,
                });
            }
        }
        q_emb_offset += qas.len();
    }

    // ---- Phase 6b: QA evaluation (parallel — IO-bound OpenAI calls). ----
    let mut qa_correct: HashMap<i64, usize> = HashMap::new();
    if qa_mode && !qa_tasks.is_empty() {
        println!("\n  running {} QA calls (50 concurrent)...", qa_tasks.len() * 2);
        let sem = Arc::new(tokio::sync::Semaphore::new(50));
        let client = Arc::new(client);
        let key = Arc::new(openai_key.unwrap());
        let mut js = tokio::task::JoinSet::new();
        for task in qa_tasks {
            let sem = sem.clone();
            let c = client.clone();
            let k = key.clone();
            js.spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let pred = answer(&c, &k, &task.question, &task.context).await?;
                let ok = judge(&c, &k, &task.question, &task.gold_answer, &pred).await?;
                Ok::<(i64, bool), anyhow::Error>((task.category, ok))
            });
        }
        let mut finished = 0usize;
        while let Some(res) = js.join_next().await {
            finished += 1;
            if finished % 100 == 0 || finished == total_questions {
                print!("\r  QA [{finished}/{total_questions}]...");
                let _ = std::io::stdout().flush();
            }
            let (cat, ok) = res??;
            if ok {
                *qa_correct.entry(cat).or_default() += 1;
                *qa_correct.entry(0).or_default() += 1;
            }
        }
    }

    // ---- Report. ----
    println!();
    let cat_name = |c: i64| match c {
        1 => "single-hop",
        2 => "multi-hop",
        3 => "temporal",
        4 => "open-domain",
        _ => "ALL",
    };
    let embed_note = if google_key.is_some() {
        format!("Google text-embedding-004 ({dim}-dim)")
    } else {
        format!("bge-small-en-v1.5 ({dim}-dim, local)")
    };
    println!("\n=== retrieval recall@{k} [{embed_note}] ===");

    println!("{:<14} {:>8} {:>8} {:>8} {:>8}  n", "category", "bm25", "vector", "fused", "full");
    for c in [1, 2, 3, 4, 0] {
        let n = *count.get(&c).unwrap_or(&0);
        if n == 0 {
            continue;
        }
        print!("{:<14}", cat_name(c));
        for mi in 0..methods.len() {
            let v = recall_sum.get(&(c, mi)).copied().unwrap_or(0.0) / n as f64;
            print!(" {:>7.1}%", v * 100.0);
        }
        println!("  {n}");
    }

    if qa_mode {
        println!("\n=== QA accuracy (gpt-5-mini answer + judge) ===");
        println!("{:<14} {:>8}  n", "category", "accuracy");
        for c in [1, 2, 3, 4, 0] {
            let n = *count.get(&c).unwrap_or(&0);
            if n == 0 {
                continue;
            }
            let acc = *qa_correct.get(&c).unwrap_or(&0) as f64 / n as f64 * 100.0;
            println!("{:<14} {:>7.1}%  {n}", cat_name(c), acc);
        }
    } else {
        println!(
            "\n({total_q} questions scored for retrieval. Re-run with --qa + OPENAI_API_KEY for QA accuracy.)"
        );
    }
    Ok(())
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
    Ok(body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .trim()
        .to_string())
}

async fn answer(
    client: &reqwest::Client,
    key: &str,
    question: &str,
    context: &str,
) -> Result<String> {
    chat(
        client,
        key,
        "Answer the question using ONLY the provided memories. Be concise: a few words or a short phrase. For dates, use the format shown in the memories.",
        &format!("Memories:\n{context}\n\nQuestion: {question}\nAnswer:"),
    )
    .await
}

async fn judge(
    client: &reqwest::Client,
    key: &str,
    question: &str,
    gold: &str,
    pred: &str,
) -> Result<bool> {
    let v = chat(
        client,
        key,
        "You are grading a model answer against a reference answer. Reply with exactly one word: CORRECT or WRONG.",
        &format!("Question: {question}\nReference answer: {gold}\nModel answer: {pred}\nIs the model answer correct (same meaning as the reference)?"),
    )
    .await?;
    Ok(v.to_uppercase().contains("CORRECT"))
}
