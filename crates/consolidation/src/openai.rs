//! OpenAI consolidator — async port of the M0 spike_llm.rs.
//!
//! Uses `json_object` response format so the model is forced into JSON.
//! The structured output shape matches `ConsolidatedMemory`:
//!
//!   { "topic_path": "...", "title": "...", "body": "...",
//!     "foresight": [{"statement": "...", "expires": "ISO-date|null"}] }

use crate::{ConsolidatedMemory, ConsolidationModel, Foresight};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;

const SYSTEM_PROMPT: &str = "\
You are a memory consolidation worker. Given raw conversation messages, distil \
them into durable semantic memory plus forward-looking foresight statements. \
Respond with STRICT JSON only, shape:
{
  \"topic_path\": \"area/subarea\",
  \"title\": \"short-noun-phrase\",
  \"body\": \"the consolidated durable facts as prose\",
  \"foresight\": [
    {\"statement\": \"...\", \"expires\": \"ISO-date or null if durable\"}
  ]
}";

pub struct OpenAiConsolidator {
    pub client: reqwest::Client,
    pub api_key: String,
    pub model: String,
}

impl OpenAiConsolidator {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: "gpt-5-mini".to_string(),
        }
    }
}

impl ConsolidationModel for OpenAiConsolidator {
    fn name(&self) -> &'static str {
        "openai"
    }

    /// gpt-5-mini's context window (shared between prompt + completion).
    fn context_window(&self) -> usize {
        272_000
    }

    fn consolidate<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> Pin<Box<dyn Future<Output = Result<ConsolidatedMemory>> + Send + 'a>> {
        Box::pin(async move {
            let joined = texts
                .iter()
                .enumerate()
                .map(|(i, t)| format!("{}. {t}", i + 1))
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

            let resp = self
                .client
                .post("https://api.openai.com/v1/chat/completions")
                .bearer_auth(&self.api_key)
                .json(&req)
                .send()
                .await
                .context("OpenAI request failed")?;

            let status = resp.status();
            let body: Value = resp.json().await.context("failed to parse OpenAI response")?;
            if !status.is_success() {
                anyhow::bail!("OpenAI {status}: {body}");
            }

            let content = body["choices"][0]["message"]["content"]
                .as_str()
                .context("missing content in OpenAI response")?;

            let v: Value = serde_json::from_str(content).context("model returned invalid JSON")?;

            let foresight = v["foresight"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|item| {
                            Some(Foresight {
                                statement: item["statement"].as_str()?.to_string(),
                                expires: item["expires"].as_str().map(|s| s.to_string()),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();

            Ok(ConsolidatedMemory {
                topic_path: v["topic_path"]
                    .as_str()
                    .unwrap_or("general")
                    .to_string(),
                title: v["title"].as_str().unwrap_or("untitled").to_string(),
                body: v["body"].as_str().unwrap_or("").to_string(),
                foresight,
            })
        })
    }
}
