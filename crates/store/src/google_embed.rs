//! Shared Google/Vertex AI embedding client + on-disk embedding cache.
//!
//! Used by the LoCoMo and LongMemEval eval binaries to batch-embed large
//! turn/question sets fast (vs. one-at-a-time local fastembed).

use anyhow::Result;
use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

// ---- Embedding cache (no extra deps — raw f32 LE bytes) ----
// Format: [n: u64][dim: u64][n*dim f32 values in row-major LE]

pub fn cache_save(path: &str, embs: &[Vec<f32>]) -> Result<()> {
    let dim = embs.first().map(|v| v.len()).unwrap_or(0);
    let mut buf = Vec::with_capacity(16 + embs.len() * dim * 4);
    buf.extend_from_slice(&(embs.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(dim as u64).to_le_bytes());
    for emb in embs {
        for v in emb {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, &buf)?;
    Ok(())
}

pub fn cache_load(path: &str, expected_n: usize) -> Option<Vec<Vec<f32>>> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 16 { return None; }
    let n = u64::from_le_bytes(data[0..8].try_into().ok()?) as usize;
    let dim = u64::from_le_bytes(data[8..16].try_into().ok()?) as usize;
    if n != expected_n || dim == 0 { return None; }
    if data.len() != 16 + n * dim * 4 { return None; }
    let mut out = Vec::with_capacity(n);
    let mut off = 16usize;
    for _ in 0..n {
        let mut emb = Vec::with_capacity(dim);
        for _ in 0..dim {
            emb.push(f32::from_le_bytes(data[off..off + 4].try_into().ok()?));
            off += 4;
        }
        out.push(emb);
    }
    Some(out)
}

/// Exchange a service account JSON key for an OAuth2 bearer token + project_id.
pub async fn sa_token(client: &reqwest::Client, key_path: &str) -> Result<(String, String)> {
    #[derive(Deserialize)]
    struct SaKey {
        client_email: String,
        private_key: String,
        project_id: String,
        #[serde(default = "default_token_uri")]
        token_uri: String,
    }
    fn default_token_uri() -> String {
        "https://oauth2.googleapis.com/token".to_string()
    }
    #[derive(Serialize)]
    struct Claims {
        iss: String,
        sub: String,
        aud: String,
        iat: u64,
        exp: u64,
        scope: String,
    }

    let sa: SaKey = serde_json::from_str(&std::fs::read_to_string(key_path)?)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let claims = Claims {
        iss: sa.client_email.clone(),
        sub: sa.client_email.clone(),
        aud: sa.token_uri.clone(),
        iat: now,
        exp: now + 3600,
        scope: "https://www.googleapis.com/auth/cloud-platform".to_string(),
    };
    let jwt = encode(
        &Header::new(Algorithm::RS256),
        &claims,
        &EncodingKey::from_rsa_pem(sa.private_key.as_bytes())?,
    )?;
    let resp: Value = client
        .post(&sa.token_uri)
        .form(&[
            ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
            ("assertion", &jwt),
        ])
        .send()
        .await?
        .json()
        .await?;
    let token = resp["access_token"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("no access_token in response: {resp}"))?;
    Ok((token, sa.project_id))
}

/// Batch-embed texts via Google text-embedding-004, outputDimensionality=384.
/// Uses Vertex AI endpoint when project_id is provided (service account auth),
/// otherwise falls back to the generativelanguage endpoint (API key / user token).
/// Greedily group texts into chunks bounded by both an item cap and an
/// estimated-token budget (chars/4), so long-form turns (LongMemEval can run
/// to 30k+ chars) don't blow past Vertex's 20k-tokens/request ceiling the way
/// a flat `chunks(100)` did for short LoCoMo turns.
fn token_aware_chunks(texts: &[String]) -> Vec<&[String]> {
    const MAX_ITEMS: usize = 100;
    const TOKEN_BUDGET: usize = 15_000;
    const CHARS_PER_TOKEN: usize = 4;
    let mut chunks: Vec<&[String]> = Vec::new();
    let mut start = 0usize;
    while start < texts.len() {
        let mut end = start;
        let mut budget = 0usize;
        while end < texts.len() && end - start < MAX_ITEMS {
            let cost = texts[end].len() / CHARS_PER_TOKEN + 1;
            if end > start && budget + cost > TOKEN_BUDGET {
                break;
            }
            budget += cost;
            end += 1;
        }
        chunks.push(&texts[start..end]);
        start = end;
    }
    chunks
}

pub async fn google_embed_batch(
    client: &reqwest::Client,
    token: &str,
    project_id: Option<&str>,
    texts: &[String],
    task_type: &str,
) -> Result<Vec<Vec<f32>>> {
    let location = std::env::var("GOOGLE_LOCATION").unwrap_or_else(|_| "us-central1".to_string());
    let chunks = token_aware_chunks(texts);
    let total_chunks = chunks.len();
    let mut all: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for (ci, chunk) in chunks.into_iter().enumerate() {
        print!("\r  batch {}/{total_chunks} ({} texts)...   ", ci + 1, chunk.len());
        let _ = std::io::stdout().flush();

        let raw_bytes = if let Some(proj) = project_id {
            // Vertex AI: POST .../text-embedding-004:predict with instances[]
            let url = format!(
                "https://{location}-aiplatform.googleapis.com/v1/projects/{proj}/locations/{location}/publishers/google/models/text-embedding-004:predict"
            );
            let instances: Vec<Value> = chunk
                .iter()
                .map(|t| {
                    json!({
                        "content": t,
                        "task_type": task_type,
                        "output_dimensionality": 384
                    })
                })
                .collect();
            client
                .post(&url)
                .bearer_auth(token)
                .json(&json!({ "instances": instances }))
                .send()
                .await?
                .bytes()
                .await?
        } else {
            // Gemini generative language endpoint — API key or user token
            let reqs: Vec<Value> = chunk
                .iter()
                .map(|t| {
                    json!({
                        "model": "models/text-embedding-004",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task_type,
                        "outputDimensionality": 384
                    })
                })
                .collect();
            client
                .post("https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents")
                .bearer_auth(token)
                .json(&json!({ "requests": reqs }))
                .send()
                .await?
                .bytes()
                .await?
        };

        let body: Value = serde_json::from_slice(&raw_bytes)
            .map_err(|e| anyhow::anyhow!("JSON parse error: {e}\nraw: {}", String::from_utf8_lossy(&raw_bytes)))?;

        // Vertex AI returns { predictions: [{ embeddings: { values: [...] } }] }
        // Gemini returns    { embeddings: [{ values: [...] }] }
        let emb_list: Vec<Vec<f32>> = if project_id.is_some() {
            body["predictions"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("Vertex AI error: {body}"))?
                .iter()
                .map(|p| {
                    p["embeddings"]["values"]
                        .as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect()
                })
                .collect()
        } else {
            body["embeddings"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("Gemini error: {body}"))?
                .iter()
                .map(|e| {
                    e["values"]
                        .as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect()
                })
                .collect()
        };
        for v in emb_list {
            anyhow::ensure!(!v.is_empty(), "empty embedding in response");
            all.push(v);
        }
    }
    println!(" done ({} embeddings)", all.len());
    Ok(all)
}

/// Resolve a (bearer_token, optional_project_id) pair from env, matching the
/// convention shared by all eval binaries:
///   GOOGLE_TOKEN                  -> direct bearer token, no project (Gemini API)
///   GOOGLE_APPLICATION_CREDENTIALS -> service account JSON path (Vertex AI)
pub async fn google_auth_from_env(client: &reqwest::Client) -> Result<Option<(String, Option<String>)>> {
    if let Some(t) = std::env::var("GOOGLE_TOKEN").ok().filter(|s| !s.is_empty()) {
        return Ok(Some((t, None)));
    }
    if let Ok(path) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
        println!("authenticating with service account {path}...");
        let (token, project) = sa_token(client, &path).await?;
        println!("  project: {project}");
        return Ok(Some((token, Some(project))));
    }
    Ok(None)
}
