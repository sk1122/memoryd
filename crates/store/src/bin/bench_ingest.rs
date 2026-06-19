//! M2 ingestion benchmark. Ingests a real LoCoMo conversation through the full
//! pipeline, timing each stage, and reports aggregate + scaling-with-store-size
//! numbers. Requires the ParadeDB container running.
//!
//! Run (release for honest numbers):  cargo run -p store --release --bin bench_ingest

use anyhow::Result;
use serde_json::Value;
use store::{Config, Store, Timings};

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = Config::load("memoryd.toml")?;
    let store = Store::connect(cfg).await?;
    store.truncate().await?;

    // Pull every turn of LoCoMo conversation 0, in session+turn order.
    let data: Value = serde_json::from_str(&std::fs::read_to_string("data/locomo10.json")?)?;
    let conv = &data[0]["conversation"];
    let mut nums: Vec<u64> = conv
        .as_object()
        .unwrap()
        .keys()
        .filter_map(|k| k.strip_prefix("session_").and_then(|r| r.parse().ok()))
        .collect();
    nums.sort_unstable();
    let mut texts: Vec<String> = Vec::new();
    for n in nums {
        if let Some(turns) = conv[format!("session_{n}")].as_array() {
            for t in turns {
                if let Some(s) = t["text"].as_str() {
                    if !s.is_empty() {
                        texts.push(s.to_string());
                    }
                }
            }
        }
    }

    println!("ingesting {} turns of LoCoMo conversation 0...\n", texts.len());

    let mut all: Vec<(Timings, bool)> = Vec::with_capacity(texts.len());
    let wall = std::time::Instant::now();
    for text in &texts {
        let d = store.ingest("bench", "private", "user", text).await?;
        all.push((d.timings, d.admitted));
    }
    let wall_s = wall.elapsed().as_secs_f64();

    let n = all.len();
    let admitted = all.iter().filter(|x| x.1).count();
    println!(
        "processed {n} turns, admitted {admitted} ({:.0}%), final store size {}",
        100.0 * admitted as f64 / n as f64,
        store.count().await?
    );
    println!(
        "wall time {:.2}s  ->  {:.1} msgs/sec  ({:.1} ms/msg avg)\n",
        wall_s,
        n as f64 / wall_s,
        wall_s / n as f64 * 1e3
    );

    // Per-stage aggregate (mean / p50 / p95).
    let stage = |f: fn(&Timings) -> f64| -> (f64, f64, f64) {
        let mut v: Vec<f64> = all.iter().map(|x| f(&x.0)).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let p = |q: f64| v[((v.len() as f64 * q) as usize).min(v.len() - 1)];
        (mean, p(0.50), p(0.95))
    };
    println!("{:<10} {:>8} {:>8} {:>8}", "stage", "mean", "p50", "p95");
    for (name, f) in [
        ("embed", (|t: &Timings| t.embed_ms) as fn(&Timings) -> f64),
        ("nearest", |t| t.nearest_ms),
        ("signals", |t| t.signals_ms),
        ("insert", |t| t.insert_ms),
        ("TOTAL", |t| t.total_ms),
    ] {
        let (m, p50, p95) = stage(f);
        println!("{name:<10} {m:>7.2}ms {p50:>6.2}ms {p95:>6.2}ms");
    }

    // Scaling: mean total latency bucketed by how many turns have been processed.
    println!("\nlatency vs progress (mean total ms per 100-turn bucket):");
    let bucket = 100;
    for start in (0..n).step_by(bucket) {
        let end = (start + bucket).min(n);
        let slice = &all[start..end];
        let mean: f64 = slice.iter().map(|x| x.0.total_ms).sum::<f64>() / slice.len() as f64;
        println!("  turns {start:>4}-{:<4} {mean:>7.2}ms", end - 1);
    }

    // Machine-readable line for charting.
    let (em, ..) = stage(|t| t.embed_ms);
    let (nm, ..) = stage(|t| t.nearest_ms);
    let (sm, ..) = stage(|t| t.signals_ms);
    let (im, ..) = stage(|t| t.insert_ms);
    println!("\nCHART embed={em:.3} nearest={nm:.3} signals={sm:.3} insert={im:.3} throughput={:.1}", n as f64 / wall_s);
    Ok(())
}
