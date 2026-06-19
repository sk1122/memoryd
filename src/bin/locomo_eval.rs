//! LoCoMo evaluation for the gzip novelty signal.
//!
//! Framing: gzip novelty is the ingestion *gate*. Its job is to drop
//! redundant/noise turns WITHOUT dropping turns that questions depend on.
//! LoCoMo annotates each QA pair with `evidence` turn IDs, so we get
//! ground-truth labels for free:
//!
//!   positive (must-keep) = turn is cited as evidence for some QA pair
//!   negative (droppable) = every other dialogue turn
//!
//! We report:
//!   1. AUC of novelty vs is-evidence (separating power), with a turn-length
//!      baseline for context.
//!   2. The evidence-retention / storage-drop tradeoff across thresholds —
//!      the gate's real operating curve and the ceiling on downstream QA.
//!
//! Memory context for each turn = a recency window of the prior turns in the
//! same conversation (streaming-gate simulation; no embeddings needed). The
//! production gate uses the top-k vector-nearest memories instead — noted.
//!
//! Usage: locomo_eval [data.json] [--window N]

use memoryd::compression_novelty;
use serde_json::Value;
use std::collections::HashSet;

struct Turn {
    text: String,
    is_evidence: bool,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut path = "data/locomo10.json".to_string();
    let mut window: usize = 30;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--window" => {
                window = args.next().and_then(|s| s.parse().ok()).unwrap_or(window);
            }
            other => path = other.to_string(),
        }
    }

    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let data: Value = serde_json::from_str(&raw).expect("invalid JSON");
    let convos = data.as_array().expect("top level must be an array");

    let mut all: Vec<(f64, bool, usize)> = Vec::new(); // (novelty, is_evidence, char_len)
    let mut total_turns = 0usize;
    let mut total_evidence = 0usize;

    for convo in convos {
        // 1. Collect all evidence turn IDs referenced by this convo's QA.
        let mut evidence: HashSet<String> = HashSet::new();
        if let Some(qas) = convo.get("qa").and_then(|q| q.as_array()) {
            for qa in qas {
                if let Some(ev) = qa.get("evidence").and_then(|e| e.as_array()) {
                    for e in ev {
                        if let Some(s) = e.as_str() {
                            evidence.insert(s.to_string());
                        }
                    }
                }
            }
        }

        // 2. Collect dialogue turns in session then turn order.
        let conv = match convo.get("conversation").and_then(|c| c.as_object()) {
            Some(c) => c,
            None => continue,
        };
        // Session keys are "session_<N>"; "session_<N>_date_time" must be excluded.
        let mut session_nums: Vec<u32> = conv
            .keys()
            .filter_map(|k| k.strip_prefix("session_").and_then(|r| r.parse::<u32>().ok()))
            .collect();
        session_nums.sort_unstable();

        let mut turns: Vec<Turn> = Vec::new();
        for n in session_nums {
            let key = format!("session_{n}");
            if let Some(list) = conv.get(&key).and_then(|v| v.as_array()) {
                for t in list {
                    let text = t.get("text").and_then(|x| x.as_str()).unwrap_or("");
                    if text.is_empty() {
                        continue;
                    }
                    let id = t.get("dia_id").and_then(|x| x.as_str()).unwrap_or("");
                    turns.push(Turn {
                        text: text.to_string(),
                        is_evidence: evidence.contains(id),
                    });
                }
            }
        }

        // 3. Score each turn: memory = recency window of prior turns.
        for i in 0..turns.len() {
            let start = i.saturating_sub(window);
            let mem: Vec<String> = turns[start..i].iter().map(|t| t.text.clone()).collect();
            let novelty = compression_novelty(&turns[i].text, &mem);
            let len = turns[i].text.chars().count();
            all.push((novelty, turns[i].is_evidence, len));
            total_turns += 1;
            if turns[i].is_evidence {
                total_evidence += 1;
            }
        }
    }

    // ---- Report ----
    let novelty_auc = auc(all.iter().map(|&(n, e, _)| (n, e)));
    let length_auc = auc(all.iter().map(|&(_, e, l)| (l as f64, e)));

    let pos: Vec<f64> = all.iter().filter(|x| x.1).map(|x| x.0).collect();
    let neg: Vec<f64> = all.iter().filter(|x| !x.1).map(|x| x.0).collect();

    println!("=== LoCoMo gzip-novelty evaluation ===");
    println!("conversations      : {}", convos.len());
    println!("dialogue turns      : {total_turns}");
    println!(
        "evidence turns      : {total_evidence}  ({:.1}% of turns, the must-keep positives)",
        100.0 * total_evidence as f64 / total_turns as f64
    );
    println!("memory window       : {window} prior turns\n");

    println!(
        "mean novelty  evidence={:.3}  non-evidence={:.3}  (gap={:+.3})",
        mean(&pos),
        mean(&neg),
        mean(&pos) - mean(&neg)
    );
    println!("AUC novelty vs is-evidence : {novelty_auc:.4}");
    println!("AUC turn-length baseline   : {length_auc:.4}");
    println!("  (0.5 = no separating power; higher = novelty predicts must-keep)\n");

    println!("threshold sweep — gate admits turns with novelty >= t:");
    println!(
        "  {:>5}  {:>16}  {:>14}",
        "t", "evidence kept", "turns dropped"
    );
    for k in 1..=9 {
        let t = k as f64 / 10.0;
        let ev_kept = pos.iter().filter(|&&n| n >= t).count();
        let dropped = all.iter().filter(|x| x.0 < t).count();
        println!(
            "  {:>5.1}  {:>7} / {:<4} {:>4.0}%  {:>6} {:>4.0}%",
            t,
            ev_kept,
            total_evidence,
            100.0 * ev_kept as f64 / total_evidence as f64,
            dropped,
            100.0 * dropped as f64 / total_turns as f64,
        );
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Rank-based AUC (Mann-Whitney), tie-aware via average ranks.
fn auc<I: Iterator<Item = (f64, bool)>>(it: I) -> f64 {
    let v: Vec<(f64, bool)> = it.collect();
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].0.partial_cmp(&v[b].0).unwrap());

    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && v[idx[j + 1]].0 == v[idx[i]].0 {
            j += 1;
        }
        // 1-based average rank for the tie group [i, j].
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[idx[k]] = avg;
        }
        i = j + 1;
    }

    let n_pos = v.iter().filter(|x| x.1).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f64::NAN;
    }
    let sum_ranks_pos: f64 = (0..n).filter(|&k| v[k].1).map(|k| ranks[k]).sum();
    (sum_ranks_pos - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos as f64 * n_neg as f64)
}
