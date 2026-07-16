//! Compression-based novelty scoring + encoding salience for the ingestion gate.
//!
//! Faithful Rust ports of:
//!   - `truememory/ingest/encoding_gate.py::_compute_novelty` (novelty signal)
//!   - `truememory/ingest/encoding_salience.py::encoding_salience_d` (salience, Variant D)
//!   - `truememory/ingest/encoding_gate.py::_is_contradiction` +
//!     `truememory/ingest/markers.py::has_update_markers` (correction detector)
//!
//! Novelty: (gzip(memory + " " + fact) - gzip(memory)) / gzip(fact)
//! High ratio → novel; low ratio → redundant.
//!
//! Salience (Variant D):
//!   ≤50 chars → speech-act classifier: noise/question/commitment/correction/greeting
//!   >50 chars → L3 legacy additive scorer: length + numbers + arousal + life events
//!
//! Note on gzip faithfulness: TrueMemory uses Python `gzip.compress(..., 6)`.
//! We use flate2 at level 6 (zlib-rs backend). Both are DEFLATE-6 in a gzip
//! container; absolute byte counts differ by a handful of header bytes but the
//! ratio is stable.

use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::Write;

// ─── gzip helpers ────────────────────────────────────────────────────────────

/// Compressed length of `data` using gzip level 6 (matches TrueMemory).
pub fn gzip_len(data: &[u8]) -> usize {
    let mut enc = GzEncoder::new(Vec::new(), Compression::new(6));
    enc.write_all(data).expect("gzip write to Vec cannot fail");
    enc.finish().expect("gzip finish to Vec cannot fail").len()
}

/// Port of the encoding gate's novelty signal.
///
/// `memory_texts` are the nearest stored memories (TrueMemory joins top-10
/// vector hits with a space). Returns a score in `[0.05, 1.0]`; empty memory
/// yields maximum novelty (1.0).
pub fn compression_novelty(fact: &str, memory_texts: &[String]) -> f64 {
    let memory_text = memory_texts
        .iter()
        .filter(|s| !s.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");

    if memory_text.trim().is_empty() {
        return 1.0;
    }

    let fact_bytes = fact.as_bytes();
    let memory_bytes = memory_text.as_bytes();

    let mut combined = Vec::with_capacity(memory_bytes.len() + 1 + fact_bytes.len());
    combined.extend_from_slice(memory_bytes);
    combined.push(b' ');
    combined.extend_from_slice(fact_bytes);

    let c_memory = gzip_len(memory_bytes);
    let c_combined = gzip_len(&combined);
    let c_fact = gzip_len(fact_bytes);

    if c_fact < 10 {
        return 0.05;
    }

    let compression_cost = (c_combined as f64 - c_memory as f64) / c_fact as f64;
    compression_cost.clamp(0.0, 1.0).max(0.05)
}

// ─── salience constants (from truememory/salience.py + encoding_salience.py) ─

// _NOISE_EXACT_V23 — speech-act scorer noise set (used for short messages ≤50 chars)
const NOISE_EXACT_V23: &[&str] = &[
    "ok", "okay", "k", "kk", "yes", "yeah", "yep", "yup", "ya", "yea",
    "no", "nah", "nope", "lol", "lmao", "lmfao", "haha", "hahaha", "heh",
    "omg", "omfg", "wtf", "nice", "cool", "dope", "sick", "lit", "fire",
    "thanks", "thx", "ty", "thank you", "got it", "gotcha",
    "sounds good", "sounds great", "bet", "word", "sure", "for sure",
    "same", "mood", "idk", "idc", "np", "no problem",
    "gn", "goodnight", "good night", "gm", "good morning", "brb", "ttyl",
    "damn", "dude", "bro", "ugh", "wow", "yikes", "ooh", "oof",
    "true", "facts", "right", "exactly", "totally", "absolutely",
    "lmao dead", "im dead", "crying", "screaming",
    // Reactions to someone else's news (_NOISE_EXACT_V23 additions)
    "that's great", "thats great", "that's awesome", "thats awesome",
    "that's amazing", "thats amazing", "that's crazy", "thats crazy",
    "that's insane", "thats insane", "that's wild", "thats wild",
    "that's so cool", "thats so cool",
    "congratulations", "congrats", "happy for you", "so happy for you",
    "proud of you", "so proud of you", "good for you",
    "no way", "are you serious", "oh my god", "oh my gosh",
    "i can't believe it", "i cant believe it", "shut up",
    "that's wonderful", "thats wonderful", "that's fantastic",
    "love that", "love it", "so cool", "so sick",
    "good luck", "you got this", "go for it", "let's go", "lets go",
    "aww", "aw", "yay", "woohoo", "woo",
];

// _NOISE_EXACT — L3 legacy scorer noise set (used for long messages >50 chars)
const NOISE_EXACT: &[&str] = &[
    "ok", "okay", "k", "kk",
    "yes", "yeah", "yep", "yup", "ya", "yea",
    "no", "nah", "nope",
    "lol", "lmao", "lmfao", "haha", "hahaha", "heh",
    "omg", "omfg", "wtf",
    "nice", "cool", "dope", "sick", "lit", "fire",
    "thanks", "thx", "ty", "thank you",
    "got it", "gotcha",
    "sounds good", "sounds great",
    "bet", "word",
    "sure", "for sure",
    "same", "mood",
    "idk", "idc",
    "np", "no problem",
    "gn", "goodnight", "good night",
    "gm", "good morning",
    "brb", "ttyl",
];

// _COMMITMENT_PATTERNS from encoding_salience.py
const COMMITMENT_PATTERNS: &[&str] = &[
    "said yes", "said no", "i'm in", "we're in", "i quit", "i did it",
    "i got it", "i got in", "i made it", "i passed", "i failed",
    "we're pregnant", "i'm pregnant", "she's pregnant",
    "i'm engaged", "we're engaged", "i'm married", "we're married",
    "i enrolled", "i applied", "i submitted", "i accepted",
    "i declined", "i resigned", "i'm leaving", "i'm moving",
    "it's booked", "it's done", "it's official", "it's over",
    "i had a baby", "had a baby", "having a baby",
    "seeing someone", "broke up", "breaking up",
    "got the job", "got the offer", "got accepted", "got rejected",
    "got promoted", "got fired", "got hired", "got laid off",
    "gave my notice", "two weeks notice", "gave notice",
    "passed away", "passed on",
];

// _UPDATE_VERBS from encoding_salience.py
const UPDATE_VERBS: &[&str] = &[
    "switched", "changed", "moved", "quit", "started", "enrolled",
    "promoted", "graduated", "launched", "resigned", "transferred",
    "hired", "fired", "accepted", "declined", "submitted",
];

// _HIGH_AROUSAL from salience.py
const HIGH_AROUSAL: &[&str] = &[
    "amazing", "incredible", "devastating", "heartbreaking",
    "thrilled", "furious", "terrified", "ecstatic", "crushed",
    "panic", "emergency", "urgent", "critical", "breakthrough",
    "milestone", "promoted", "fired", "pregnant", "engaged",
    "diagnosed", "accident", "passed away", "died",
];

// _LIFE_EVENTS from salience.py
const LIFE_EVENTS: &[&str] = &[
    "got married", "got engaged", "having a baby", "got promoted",
    "got fired", "broke up", "moved to", "graduated", "launched",
    "raised funding", "demo day", "ipo", "acquisition",
];

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Check if `text` contains `word` with ASCII word-boundaries on both sides.
fn word_in(text: &str, word: &str) -> bool {
    let tb = text.as_bytes();
    let wb = word.as_bytes();
    let wn = wb.len();
    if wn == 0 || text.len() < wn {
        return false;
    }
    for i in 0..=(text.len() - wn) {
        if &tb[i..i + wn] == wb {
            let before_ok = i == 0 || !tb[i - 1].is_ascii_alphanumeric();
            let after_ok = i + wn >= tb.len() || !tb[i + wn].is_ascii_alphanumeric();
            if before_ok && after_ok {
                return true;
            }
        }
    }
    false
}

/// Port of `_COMMITMENT_RE` — "i <action-verb>" first-person patterns.
fn is_commitment_like(lower: &str) -> bool {
    // "i <verb>" — past-tense first-person action statements
    const I_VERBS: &[&str] = &[
        "got", "did", "made", "found", "built", "started", "quit", "left",
        "joined", "enrolled", "accepted", "submitted", "finished", "completed",
        "signed", "bought", "sold", "moved", "said", "told", "asked", "proposed",
        "created", "launched", "shipped", "published", "passed", "graduated",
        "earned", "won", "lost", "broke", "fixed",
    ];
    for v in I_VERBS {
        // Build "i <verb>" and check with word boundaries on both sides
        let mut phrase = String::with_capacity(v.len() + 2);
        phrase.push_str("i ");
        phrase.push_str(v);
        if word_in(lower, &phrase) {
            return true;
        }
    }

    // "i'm <state>" patterns
    const IM_STATES: &[&str] = &[
        "i'm pregnant", "i'm engaged", "i'm leaving", "i'm moving",
        "i'm starting", "i'm quitting", "i'm going to", "i'm seeing someone",
        "i'm having a",
    ];
    if IM_STATES.iter().any(|p| lower.contains(p)) {
        return true;
    }

    // "we're <state>" patterns
    const WERE_STATES: &[&str] = &[
        "we're pregnant", "we're engaged", "we're moving",
        "we're having", "we're getting", "we're doing",
    ];
    if WERE_STATES.iter().any(|p| lower.contains(p)) {
        return true;
    }

    // remaining high-commitment phrases
    lower.contains("i gave my notice")
        || lower.contains("i gave notice")
        || lower.contains("it's booked")
        || lower.contains("it's official")
        || lower.contains("it's confirmed")
        || lower.contains("it's done")
        || lower.contains("it's over")
        || lower.contains("it's happening")
}

/// Port of TrueMemory's `_speech_act_score()` — classifies short messages by
/// linguistic function (speech act) rather than content. Length-independent.
fn speech_act_score(text: &str) -> f64 {
    let lower = text.to_lowercase();
    let lower = lower.trim();

    // Noise
    if NOISE_EXACT_V23.iter().any(|&n| n == lower) {
        return 0.02;
    }

    // Question: ends with "?" or starts with a question word
    if text.trim().ends_with('?')
        || lower.starts_with("what ")
        || lower.starts_with("how ")
        || lower.starts_with("why ")
        || lower.starts_with("where ")
        || lower.starts_with("when ")
        || lower.starts_with("who ")
        || lower.starts_with("which ")
        || lower.starts_with("do you")
        || lower.starts_with("are you")
        || lower.starts_with("is it")
        || lower.starts_with("can you")
        || lower.starts_with("could you")
    {
        return 0.2;
    }

    // Commitment regex (high-confidence first-person announcements)
    if is_commitment_like(lower) {
        return 0.8;
    }

    // Commitment phrase set
    if COMMITMENT_PATTERNS.iter().any(|p| lower.contains(p)) {
        return 0.7;
    }

    // Correction / update language
    if lower.contains("no longer")
        || lower.contains("not anymore")
        || word_in(lower, "instead")
        || lower.contains("correction")
        || UPDATE_VERBS.iter().any(|v| word_in(lower, v))
        || (lower.contains("actually") && lower.contains(" not "))
    {
        return 0.6;
    }

    // Greeting
    if lower.starts_with("hey ")
        || lower == "hey"
        || lower.starts_with("hi ")
        || lower == "hi"
        || lower.starts_with("hello")
        || lower.starts_with("yo ")
        || lower == "yo"
        || lower.starts_with("sup")
        || lower.starts_with("what's up")
        || lower.starts_with("howdy")
    {
        return 0.05;
    }

    // Laugh / filler exclamation
    if lower.starts_with("haha")
        || lower.starts_with("lol")
        || lower.starts_with("lmao")
        || lower.starts_with("omg")
        || lower.starts_with("wow")
        || lower.starts_with("damn")
        || lower.starts_with("ugh")
        || lower.starts_with("yikes")
    {
        return 0.08;
    }

    // 5+ alphabetic words → substantive short message
    let word_count = lower
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|s| !s.is_empty())
        .count();
    if word_count >= 5 {
        return 0.5;
    }

    0.25
}

/// Port of TrueMemory's `_score_legacy()` from `truememory/salience.py` —
/// additive heuristic scorer for messages longer than 50 chars.
fn score_legacy(text: &str) -> f64 {
    let t = text.trim();
    let lower = t.to_lowercase();
    let norm: &str = lower.trim_matches(|c: char| "!?.… ".contains(c));

    let mut score = 0.30_f64;

    // Noise penalty
    if NOISE_EXACT.iter().any(|&n| n == norm) {
        score -= 0.30;
    }

    // Length bonus (log-scaled, caps at ~200 chars)
    let length = t.len();
    match length {
        0..=9 => score -= 0.10,
        10..=29 => {}
        30..=99 => score += 0.10,
        100..=199 => score += 0.20,
        _ => score += 0.25,
    }

    // Number / money / date bonus
    let has_money = t.as_bytes().windows(2).any(|w| w[0] == b'$' && w[1].is_ascii_digit());
    let has_numbers = t.chars().any(|c| c.is_ascii_digit());
    const MONTHS: &[&str] = &[
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec",
    ];
    let has_date = MONTHS.iter().any(|m| {
        if let Some(pos) = lower.find(m) {
            lower[pos + m.len()..]
                .trim_start()
                .chars()
                .next()
                .map_or(false, |c| c.is_ascii_digit())
        } else {
            false
        }
    });

    if has_money {
        score += 0.15;
    } else if has_numbers || has_date {
        score += 0.10;
    }

    // Structural content: newlines + bullet points
    if t.contains('\n') && length > 50 {
        score += 0.05;
    }
    if t.lines().any(|line| {
        let l = line.trim_start();
        l.starts_with("- ") || l.starts_with("* ") || l.starts_with("• ")
    }) {
        score += 0.05;
    }

    // Exclamation density
    let excl = t.chars().filter(|&c| c == '!').count();
    if excl >= 3 {
        score += 0.15;
    } else if excl >= 1 {
        score += 0.05;
    }

    // ALL-CAPS words (≥2 chars, at least one cased char, no lowercase)
    let caps_count = t
        .split_whitespace()
        .filter(|w| {
            w.len() > 1
                && w.chars().any(|c| c.is_alphabetic())
                && w.chars().all(|c| !c.is_lowercase())
        })
        .count();
    score += (caps_count as f64 * 0.05).min(0.10);

    // High-arousal vocabulary
    let arousal_hits = HIGH_AROUSAL.iter().filter(|&&w| lower.contains(w)).count();
    score += (arousal_hits as f64 * 0.10).min(0.20);

    // Life-event phrases
    let event_hits = LIFE_EVENTS.iter().filter(|&&e| lower.contains(e)).count();
    score += (event_hits as f64 * 0.15).min(0.30);

    score.clamp(0.0, 1.0)
}

// ─── public API ──────────────────────────────────────────────────────────────

/// Encoding-importance salience: "is this worth remembering?" — LLM-free.
///
/// Port of TrueMemory's `encoding_salience_d()` (Variant D):
///   ≤50 chars → `_speech_act_score`: classifies by linguistic function
///   >50 chars → `_score_legacy`: L3 additive heuristic scorer
///
/// Returns [0, 1].
pub fn rule_salience(text: &str) -> f64 {
    let t = text.trim();
    if t.is_empty() {
        return 0.0;
    }
    if t.len() <= 50 {
        speech_act_score(t)
    } else {
        score_legacy(t)
    }
}

/// Heuristic correction/contradiction detector.
///
/// Port of TrueMemory's `_is_contradiction()` from `encoding_gate.py` plus
/// `has_update_markers()` from `ingest/markers.py`. Corrections bypass the
/// encoding gate regardless of novelty/salience (issue #585), so this errs
/// toward recall. LLM-free; the cold path does real arbitration later.
pub fn is_correction(text: &str) -> bool {
    let lower = text.to_lowercase();

    // "not X but Y" structural pattern
    if lower.contains(" not ") && lower.contains(" but ") {
        return true;
    }

    // UPDATE_MARKERS from truememory/ingest/markers.py (plain-substring markers)
    // Word-boundary matching via word_in() for single words; substring for phrases.
    if word_in(&lower, "actually")
        || lower.contains("correction:")
        || lower.contains("correction -")
        || lower.contains("no longer")
        || lower.contains("not anymore")
        || lower.contains("changed to")
        || lower.contains("changed from")
        || lower.contains("switched to")
        || lower.contains("switched from")
        || lower.contains("moved to")
        || lower.contains("used to be")
        || lower.contains("used to")
        || lower.contains("instead of")
        || lower.contains("wrong about")
        || lower.contains("was wrong")
        || lower.contains("is wrong")
        || lower.contains("not true")
        || lower.contains("isn't true")
        || lower.contains("that's incorrect")
        || lower.contains("that is incorrect")
        || word_in(&lower, "updated")
        || word_in(&lower, "replaced")
        || word_in(&lower, "formerly")
        || word_in(&lower, "previously")
    {
        return true;
    }

    // Structural patterns from markers.py
    // "now is/uses/prefers/lives/works/takes/runs/has"
    const NOW_VERBS: &[&str] = &[
        "now is ", "now use", "now prefer", "now live",
        "now work", "now take", "now run", "now has",
    ];
    if NOW_VERBS.iter().any(|v| lower.contains(v)) {
        return true;
    }

    // "was ... now" (fact changed over time)
    if lower.contains(" was ") && lower.contains(" now ") {
        return true;
    }

    // Direction/number-change arrows
    if lower.contains(" -> ") || lower.contains(" => ") || lower.contains(" --> ") {
        return true;
    }

    // Temporal-change phrases
    lower.contains("since ")
        || lower.contains("as of ")
        || lower.contains("starting ")
        || lower.contains("effective ")
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn salience_floors_noise() {
        // Pure backchannel — noise set → 0.02
        assert!(rule_salience("ok") < 0.1);
        assert!(rule_salience("thanks") < 0.1);
        // Congratulatory reactions → noise set (V23)
        assert!(rule_salience("congrats") < 0.1);
        assert!(rule_salience("that's great") < 0.1);
        // Substantive message — 9 alphabetic words → 0.5
        assert!(rule_salience("I rotated the signing secret to v9 last sprint") > 0.3);
    }

    #[test]
    fn salience_speech_act_tiers() {
        // Commitment → 0.8
        assert_eq!(rule_salience("I got the job"), 0.8);
        // Commitment pattern → 0.7
        assert_eq!(rule_salience("got promoted"), 0.7);
        // Correction → 0.6
        assert_eq!(rule_salience("I switched to Rust"), 0.6);
        // Question → 0.2
        assert_eq!(rule_salience("What did you do today?"), 0.2);
        // Greeting → 0.05
        assert!(rule_salience("Hey how are you") <= 0.05 + 0.01);
    }

    #[test]
    fn salience_long_message() {
        // Long message with numbers + life event → well above 0.3
        let msg = "I just got promoted to senior engineer with a $180k salary and relocation";
        assert!(rule_salience(msg) > 0.5);
    }

    #[test]
    fn detects_corrections() {
        assert!(is_correction("Actually, I rotated the JWT secret"));
        assert!(is_correction("It's not Google but Meta now"));
        assert!(is_correction("I switched to Rust last month"));
        assert!(is_correction("Previously I was at Google, now I work at Anthropic"));
        assert!(!is_correction("We adopted a puppy named Scout"));
    }

    // Mirror TrueMemory's test_encoding_gate_compression_novelty.py invariants.

    #[test]
    fn empty_memory_is_maximally_novel() {
        let n = compression_novelty("I just got a new job at Google", &[]);
        assert_eq!(n, 1.0);
    }

    #[test]
    fn redundant_message_scores_low() {
        let mem = vec!["I work at Google as a software engineer".to_string()];
        let n = compression_novelty("I work at Google as a software engineer", &mem);
        assert!(n < 0.5, "redundant should be low novelty, got {n:.3}");
    }

    #[test]
    fn novel_beats_restatement() {
        let mem = vec!["I work at Google as a software engineer".to_string()];
        let restated = compression_novelty("I work at Google as a software engineer", &mem);
        let novel = compression_novelty("We just adopted a golden retriever puppy named Scout", &mem);
        assert!(novel > 0.4, "novel content should be substantially novel, got {novel:.3}");
        assert!(
            novel > restated + 0.2,
            "novel {novel:.3} should clearly exceed restatement {restated:.3}"
        );
    }

    #[test]
    fn noise_scores_low() {
        let mem = vec![
            "I work at Google".to_string(),
            "We moved to Portland last month".to_string(),
        ];
        let n = compression_novelty("ok", &mem);
        assert!(n < 0.3, "noise 'ok' should score < 0.3, got {n:.3}");
    }

    #[test]
    fn signal_beats_noise() {
        let mem = vec![
            "I work at Google".to_string(),
            "We moved to Portland".to_string(),
            "The salary is $150,000".to_string(),
        ];
        let noise = compression_novelty("ok", &mem);
        let signal = compression_novelty("I just got engaged to Riley last weekend", &mem);
        assert!(signal > noise, "signal {signal:.3} must beat noise {noise:.3}");
    }

    #[test]
    fn scores_stay_in_range() {
        let mem = vec!["Some existing memory about work and life".to_string()];
        for msg in ["ok", "I got a new job", &"A".repeat(500), "", "🎉🎉🎉"] {
            let n = compression_novelty(msg, &mem);
            assert!((0.0..=1.0).contains(&n), "out of range for {msg:?}: {n}");
        }
    }
}
