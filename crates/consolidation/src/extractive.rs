//! Extractive consolidator — heuristic, fully local, zero extra deps.
//!
//! Does NOT use an LLM. Instead:
//!   title       → first sentence, capped at 80 chars
//!   topic_path  → keyword scan over a fixed taxonomy
//!   body        → input joined + light deduplication
//!   foresight   → sentences with future-tense markers

use crate::{ConsolidatedMemory, ConsolidationModel, Foresight};
use anyhow::Result;
use std::future::Future;
use std::pin::Pin;

pub struct ExtractiveConsolidator;

impl ConsolidationModel for ExtractiveConsolidator {
    fn name(&self) -> &'static str {
        "extractive"
    }

    /// No real API limit (local, CPU-only) — cap generously so a cluster
    /// still stays a bounded, finite chunk of work.
    fn context_window(&self) -> usize {
        1_000_000
    }

    fn consolidate<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> Pin<Box<dyn Future<Output = Result<ConsolidatedMemory>> + Send + 'a>> {
        let all = texts.join(" ");
        Box::pin(async move {
            Ok(ConsolidatedMemory {
                title: extract_title(&all),
                topic_path: detect_topic(&all),
                body: dedup_body(&all),
                foresight: extract_foresight(&all),
            })
        })
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn extract_title(text: &str) -> String {
    let sentence = text
        .split(['.', '!', '?'])
        .find(|s| s.trim().split_whitespace().count() >= 3)
        .unwrap_or(text);
    let t = sentence.trim();
    if t.len() <= 80 {
        t.to_string()
    } else {
        // Break at last word boundary before 80 chars
        let cut = t[..80]
            .rfind(' ')
            .unwrap_or(80.min(t.len()));
        format!("{}…", &t[..cut])
    }
}

fn detect_topic(text: &str) -> String {
    let l = text.to_lowercase();

    // Engineering / tech
    if l.contains("engineer") || l.contains("software") || l.contains("deploy")
        || l.contains("code") || l.contains("api") || l.contains("server")
        || l.contains("database") || l.contains("bug") || l.contains("feature")
    {
        return "work/engineering".to_string();
    }
    // General work
    if l.contains("job") || l.contains("work") || l.contains("career")
        || l.contains("office") || l.contains("promotion") || l.contains("manager")
        || l.contains("company") || l.contains("salary")
    {
        return "work/general".to_string();
    }
    // Finance
    if l.contains("money") || l.contains("invest") || l.contains("bank")
        || l.contains("funding") || l.contains("revenue") || l.contains('$')
        || l.contains("loan") || l.contains("rent")
    {
        return "finance".to_string();
    }
    // Health
    if l.contains("health") || l.contains("doctor") || l.contains("hospital")
        || l.contains("medication") || l.contains("diagnosis") || l.contains("symptom")
        || l.contains("exercise") || l.contains("diet")
    {
        return "health".to_string();
    }
    // Relationships / social
    if l.contains("partner") || l.contains("friend") || l.contains("family")
        || l.contains("married") || l.contains("relationship") || l.contains("baby")
        || l.contains("engaged") || l.contains("broke up")
    {
        return "personal/relationships".to_string();
    }
    // Education
    if l.contains("school") || l.contains("university") || l.contains("degree")
        || l.contains("graduate") || l.contains("study") || l.contains("course")
    {
        return "education".to_string();
    }
    // Travel / location
    if l.contains("travel") || l.contains("moved to") || l.contains("moving to")
        || l.contains("flight") || l.contains("visa") || l.contains("city")
    {
        return "life/location".to_string();
    }

    "general".to_string()
}

fn dedup_body(text: &str) -> String {
    // Split into sentences, lowercase-deduplicate, rejoin.
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for sentence in text.split(['.', '!', '?']) {
        let s = sentence.trim();
        if s.is_empty() || s.split_whitespace().count() < 3 {
            continue;
        }
        let key = s.to_lowercase();
        if seen.insert(key) {
            out.push(s);
        }
    }
    out.join(". ")
}

fn extract_foresight(text: &str) -> Vec<Foresight> {
    const FUTURE_MARKERS: &[&str] = &[
        "will ", "going to ", "planning to ", "plan to ", "intend to ",
        "next week", "next month", "next year", "soon", "eventually",
        "hope to", "want to", "would like to", "expect to",
    ];

    let mut out = Vec::new();
    for sentence in text.split(['.', '!', '?']) {
        let s = sentence.trim();
        if s.split_whitespace().count() < 4 {
            continue;
        }
        let lower = s.to_lowercase();
        if FUTURE_MARKERS.iter().any(|m| lower.contains(m)) {
            out.push(Foresight {
                statement: s.to_string(),
                expires: None,
            });
            if out.len() == 3 {
                break;
            }
        }
    }
    out
}
