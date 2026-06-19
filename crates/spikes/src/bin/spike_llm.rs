//! M0 Spike 3: consolidation LLM (the cold path).
//!
//! Mirrors the real M5 task: hand the model a cluster of raw messages and ask
//! for STRUCTURED consolidation output — a semantic summary plus foresight
//! statements with validity. The risk we're de-risking is structured-output
//! reliability, which is exactly what M5 depends on.
//!
//! The model sits behind a `ConsolidationModel` trait so swapping the hosted
//! OpenAI model for local Gemma-3-4B-via-Ollama later is a single new impl.
//!
//! Run with:  OPENAI_API_KEY=sk-... cargo run -p spikes --bin spike_llm

use serde_json::{json, Value};

trait ConsolidationModel {
    /// Return strict JSON consolidating the given raw messages.
    fn consolidate(&self, messages: &[&str]) -> anyhow::Result<Value>;
}

struct OpenAi {
    api_key: String,
    model: String,
}

const SYSTEM_PROMPT: &str = "\
You are a memory consolidation worker. Given raw conversation messages, distill \
them into durable semantic memory plus forward-looking 'foresight' statements. \
Respond with STRICT JSON only, shape:
{
  \"topic_path\": \"area/subarea\",
  \"title\": \"short-noun-phrase\",
  \"body\": \"the consolidated durable facts as prose\",
  \"foresight\": [
    {\"statement\": \"...\", \"expires\": \"ISO-date or null if durable\"}
  ]
}";

impl ConsolidationModel for OpenAi {
    fn consolidate(&self, messages: &[&str]) -> anyhow::Result<Value> {
        let joined = messages
            .iter()
            .enumerate()
            .map(|(i, m)| format!("{}. {m}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");

        let req = json!({
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format!("Consolidate these messages:\n{joined}")}
            ]
        });

        let client = reqwest::blocking::Client::new();
        let resp = client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&req)
            .send()?;

        let status = resp.status();
        let body: Value = resp.json()?;
        if !status.is_success() {
            anyhow::bail!("OpenAI {status}: {body}");
        }
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("no message content in response: {body}"))?;
        // The model claims JSON; parsing it here IS the structured-output test.
        Ok(serde_json::from_str(content)?)
    }
}

fn main() -> anyhow::Result<()> {
    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("OPENAI_API_KEY not set. Run:\n  OPENAI_API_KEY=sk-... cargo run -p spikes --bin spike_llm");
            std::process::exit(2);
        }
    };

    let model: Box<dyn ConsolidationModel> = Box::new(OpenAi {
        api_key,
        model: "gpt-5-mini".to_string(),
    });

    // A small raw cluster: a durable fact + a transient state (should become foresight w/ expiry).
    let messages = [
        "I switched our auth library to jsonwebtoken v9.",
        "Right now I'm mid-migration, still rotating the old signing secrets this sprint.",
        "Long term we're standardizing all services on jsonwebtoken v9.",
    ];

    let out = model.consolidate(&messages)?;
    println!("{}", serde_json::to_string_pretty(&out)?);

    // Validate the structure M5 will rely on.
    let ok = out.get("topic_path").is_some()
        && out.get("title").is_some()
        && out.get("body").is_some()
        && out.get("foresight").map_or(false, |f| f.is_array());
    if ok {
        println!("\nSPIKE PASS: well-formed structured consolidation output.");
    } else {
        anyhow::bail!("structured output missing required fields");
    }
    Ok(())
}
