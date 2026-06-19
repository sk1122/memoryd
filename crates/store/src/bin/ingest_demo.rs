//! M2 end-to-end demo: ingest a scripted sequence through the real pipeline
//! (embed -> vector-nearest context -> gzip novelty + salience -> insert),
//! and assert the gate behaves. Requires the ParadeDB container running.
//!
//! Run:  cargo run -p store --bin ingest_demo

use anyhow::Result;
use store::{Config, Store};

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = Config::load("memoryd.toml")?;
    let store = Store::connect(cfg).await?;
    store.truncate().await?; // clean slate for the demo

    // (text, expected_admit) — None means "report only, don't assert".
    let script: &[(&str, Option<bool>)] = &[
        ("I work at Google as a staff software engineer.", Some(true)), // 1 empty mem -> novel
        ("We adopted a golden retriever puppy named Scout.", Some(true)), // 2 new topic
        ("My team standardized on jsonwebtoken v9 for auth.", Some(true)), // 3 new topic
        ("ok", Some(false)),                                            // 4 noise -> floor
        ("I work at Google as a staff software engineer.", Some(false)), // 5 exact dup -> low novelty
        ("Actually, I rotated the JWT signing secret to v9 last sprint.", Some(true)), // 6 correction
        ("thanks", Some(false)),                                        // 7 noise -> floor
        (
            "Scout had his first vet appointment Tuesday and needs a booster shot in three weeks.",
            None, // 8 new info about an existing topic — the case cosine would miss
        ),
    ];

    println!(
        "{:<3} {:<6} {:>7} {:>8}  {:<22} message",
        "#", "admit", "novelty", "salience", "reason"
    );
    let mut failures = 0;
    for (i, (text, expect)) in script.iter().enumerate() {
        let d = store.ingest("demo", "private", "user", text).await?;
        let mark = match expect {
            Some(e) if *e == d.admitted => "ok",
            Some(_) => {
                failures += 1;
                "MISMATCH"
            }
            None => "·",
        };
        println!(
            "{:<3} {:<6} {:>7.3} {:>8.3}  {:<22} {} {}",
            i + 1,
            if d.admitted { "ADMIT" } else { "drop" },
            d.novelty,
            d.salience,
            d.reason,
            mark,
            truncate(text, 50),
        );
    }

    let n = store.count().await?;
    println!("\nrows stored: {n}");
    if failures == 0 {
        println!("M2 PASS: gate behaved as expected end-to-end (embed -> NN context -> gate -> store).");
        Ok(())
    } else {
        anyhow::bail!("{failures} gate decision(s) did not match expectation");
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}
