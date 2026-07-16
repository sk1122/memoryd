//! M2: storage core + ingestion gate.
//!
//! Owns the Postgres schema, embeds on write, and runs the LLM-free gate:
//!   1. embed the incoming message (bge-small, 384-dim)
//!   2. fetch the k nearest stored memories (scoped to the agent + shared)
//!   3. compute novelty (gzip, against those real neighbors) + salience
//!   4. admit/drop; corrections always bypass
//!   5. on admit, insert the row with its embedding
//!
//! This is the first time the M1 novelty signal meets the real retrieval path:
//! the compression context is now actual vector-nearest memory, not a recency
//! window.

pub mod google_embed;

use anyhow::Result;
use sqlx::AssertSqlSafe;
use fastembed::{
    EmbeddingModel, InitOptions, RerankInitOptions, RerankerModel, TextEmbedding, TextRerank,
};
use signals::{compression_novelty, is_correction, rule_salience};
use serde::Deserialize;
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// A retrieved memory with its final relevance score.
#[derive(Debug, Clone)]
pub struct Hit {
    pub id: i64,
    pub text: String,
    pub score: f64,
}

/// A stored memory row returned by `Store::list`.
#[derive(Debug)]
pub struct MemoryRow {
    pub id: i64,
    pub scope: String,
    pub role: String,
    pub text: String,
    pub ts: i64,
    pub novelty: Option<f32>,
    pub salience: Option<f32>,
}

/// Per-agent memory statistics returned by `Store::profile`.
#[derive(Debug)]
pub struct Profile {
    pub agent_id: String,
    pub total: i64,
    pub private: i64,
    pub shared: i64,
    pub avg_novelty: f64,
    pub avg_salience: f64,
    pub oldest_ts: Option<i64>,
    pub newest_ts: Option<i64>,
}

/// Reciprocal Rank Fusion of several ranked id-lists. K=60 is the standard.
pub fn rrf(lists: &[Vec<i64>]) -> Vec<i64> {
    const K: f64 = 60.0;
    let mut score: HashMap<i64, f64> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            *score.entry(*id).or_default() += 1.0 / (K + rank as f64 + 1.0);
        }
    }
    let mut ids: Vec<i64> = score.keys().copied().collect();
    ids.sort_by(|a, b| score[b].partial_cmp(&score[a]).unwrap());
    ids
}

#[derive(Debug, Deserialize)]
pub struct Config {
    pub database: Database,
    pub gate: Gate,
}

#[derive(Debug, Deserialize)]
pub struct Database {
    pub url: String,
}

#[derive(Debug, Deserialize)]
pub struct Gate {
    pub novelty_threshold: f64,
    pub salience_floor: f64,
    pub neighbor_k: i64,
}

impl Config {
    pub fn load(path: &str) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&raw)?)
    }
}

/// Per-stage latencies for one ingest, in milliseconds.
#[derive(Debug, Default, Clone, Copy)]
pub struct Timings {
    pub embed_ms: f64,
    pub nearest_ms: f64,
    pub signals_ms: f64,
    pub insert_ms: f64,
    pub total_ms: f64,
}

/// The outcome of passing one message through the gate.
#[derive(Debug)]
pub struct Decision {
    pub admitted: bool,
    pub id: Option<i64>,
    pub novelty: f64,
    pub salience: f64,
    pub correction: bool,
    pub reason: &'static str,
    pub timings: Timings,
}

pub struct Store {
    pool: PgPool,
    model: TextEmbedding,
    reranker: TextRerank,
    gate: Gate,
}

impl Store {
    pub async fn connect(cfg: Config) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&cfg.database.url)
            .await?;
        let model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::BGESmallENV15))?;
        let reranker = TextRerank::try_new(RerankInitOptions::new(RerankerModel::BGERerankerBase))?;
        let store = Self {
            pool,
            model,
            reranker,
            gate: cfg.gate,
        };
        store.migrate().await?;
        Ok(store)
    }

    /// Idempotent schema. The daemon owns its tables (not docker init.sql).
    ///
    /// NOTE: hardcodes VECTOR(384) for BGESmallENV15. If you change the
    /// embedding model or dimension, call `reset_for_dim(dim)` after connecting
    /// to drop and recreate the table with the correct vector width. The
    /// `IF NOT EXISTS` guards here mean a wrong-dim table will silently persist.
    async fn migrate(&self) -> Result<()> {
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&self.pool)
            .await?;
        sqlx::query("CREATE EXTENSION IF NOT EXISTS pg_search")
            .execute(&self.pool)
            .await?;
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS messages (
                id        BIGSERIAL PRIMARY KEY,
                agent_id  TEXT NOT NULL,
                scope     TEXT NOT NULL,          -- 'private' | 'shared'
                role      TEXT NOT NULL,          -- 'user' | 'assistant' | 'system'
                text      TEXT NOT NULL,
                ts        BIGINT NOT NULL,        -- unix millis
                novelty   REAL,
                salience  REAL,
                embedding VECTOR(384)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS messages_embedding_hnsw \
             ON messages USING hnsw (embedding vector_cosine_ops)",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS messages_bm25 \
             ON messages USING bm25 (id, text) WITH (key_field = 'id')",
        )
        .execute(&self.pool)
        .await?;

        // M5: consolidations table — one row per (message, model).
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS consolidations (
                id          BIGSERIAL PRIMARY KEY,
                message_id  BIGINT NOT NULL,
                topic_path  TEXT NOT NULL,
                title       TEXT NOT NULL,
                body        TEXT NOT NULL,
                foresight   TEXT NOT NULL,   -- JSON array
                model       TEXT NOT NULL,
                ts          BIGINT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE UNIQUE INDEX IF NOT EXISTS consolidations_msg_model \
             ON consolidations (message_id, model)",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = self.model.embed(vec![text.to_string()], None)?;
        Ok(v.remove(0))
    }

    /// BM25 lexical search (ParadeDB pg_search), scoped to one agent + shared.
    pub async fn bm25_search(
        &self,
        agent_id: &str,
        query: &str,
        limit: i64,
    ) -> Result<Vec<(i64, String)>> {
        let rows: Vec<(i64, String)> = sqlx::query_as(
            "SELECT id, text FROM messages \
             WHERE (agent_id = $1 OR scope = 'shared') AND id @@@ paradedb.match('text', $2) \
             ORDER BY paradedb.score(id) DESC LIMIT $3",
        )
        .bind(agent_id)
        .bind(query)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Dense vector search (pgvector cosine), scoped to one agent + shared.
    pub async fn vector_search(
        &self,
        agent_id: &str,
        emb: &[f32],
        limit: i64,
    ) -> Result<Vec<(i64, String)>> {
        let v = pgvector::Vector::from(emb.to_vec());
        let rows: Vec<(i64, String)> = sqlx::query_as(
            "SELECT id, text FROM messages \
             WHERE (agent_id = $1 OR scope = 'shared') \
             ORDER BY embedding <=> $2 LIMIT $3",
        )
        .bind(agent_id)
        .bind(&v)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Cross-encoder rerank a candidate set, returning the top_k hits.
    pub fn rerank(&self, query: &str, candidates: &[(i64, String)], top_k: usize) -> Result<Vec<Hit>> {
        if candidates.is_empty() {
            return Ok(vec![]);
        }
        let docs: Vec<&str> = candidates.iter().map(|(_, t)| t.as_str()).collect();
        let results = self.reranker.rerank(query, docs, false, None)?;
        let mut hits: Vec<Hit> = results
            .into_iter()
            .take(top_k)
            .map(|r| {
                let (id, text) = &candidates[r.index];
                Hit {
                    id: *id,
                    text: text.clone(),
                    score: r.score as f64,
                }
            })
            .collect();
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Full retrieval pipeline: BM25 + dense -> RRF -> cross-encoder rerank.
    pub async fn search(&self, agent_id: &str, query: &str, top_k: usize) -> Result<Vec<Hit>> {
        let emb = self.embed(query)?;
        let bm = self.bm25_search(agent_id, query, 100).await?;
        let vec = self.vector_search(agent_id, &emb, 100).await?;

        let mut text: HashMap<i64, String> = HashMap::new();
        for (id, t) in bm.iter().chain(vec.iter()) {
            text.entry(*id).or_insert_with(|| t.clone());
        }
        let fused = rrf(&[
            bm.iter().map(|(id, _)| *id).collect(),
            vec.iter().map(|(id, _)| *id).collect(),
        ]);
        let cand: Vec<(i64, String)> = fused
            .into_iter()
            .take(50)
            .map(|id| (id, text[&id].clone()))
            .collect();
        self.rerank(query, &cand, top_k)
    }

    /// Insert a message with a pre-computed embedding (no fastembed call).
    /// Used by the benchmark when embeddings are pre-batched via a remote API.
    pub async fn store_raw_vec(
        &self,
        agent_id: &str,
        scope: &str,
        role: &str,
        text: &str,
        emb: Vec<f32>,
    ) -> Result<i64> {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let v = pgvector::Vector::from(emb);
        let id: i64 = sqlx::query_scalar(
            "INSERT INTO messages (agent_id, scope, role, text, ts, embedding) \
             VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        )
        .bind(agent_id)
        .bind(scope)
        .bind(role)
        .bind(text)
        .bind(ts)
        .bind(&v)
        .fetch_one(&self.pool)
        .await?;
        Ok(id)
    }

    /// Insert a message unconditionally (no gate). Used to load the episodic
    /// substrate for the retrieval benchmark, where the gate is out of scope.
    pub async fn store_raw(
        &self,
        agent_id: &str,
        scope: &str,
        role: &str,
        text: &str,
    ) -> Result<i64> {
        let emb = self.embed(text)?;
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let v = pgvector::Vector::from(emb);
        let id: i64 = sqlx::query_scalar(
            "INSERT INTO messages (agent_id, scope, role, text, ts, embedding) \
             VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        )
        .bind(agent_id)
        .bind(scope)
        .bind(role)
        .bind(text)
        .bind(ts)
        .bind(&v)
        .fetch_one(&self.pool)
        .await?;
        Ok(id)
    }

    /// Texts of the k nearest stored memories, scoped: the agent's own
    /// private memory plus anything shared. This `agent_id = :me OR scope =
    /// 'shared'` filter IS the multi-agent isolation mechanism.
    async fn nearest(&self, agent_id: &str, emb: &[f32], k: i64) -> Result<Vec<String>> {
        let v = pgvector::Vector::from(emb.to_vec());
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT text FROM messages \
             WHERE (agent_id = $1 OR scope = 'shared') \
             ORDER BY embedding <=> $2 LIMIT $3",
        )
        .bind(agent_id)
        .bind(&v)
        .bind(k)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(|(t,)| t).collect())
    }

    /// Pass one message through the gate; insert it iff admitted.
    pub async fn ingest(
        &self,
        agent_id: &str,
        scope: &str,
        role: &str,
        text: &str,
    ) -> Result<Decision> {
        let t_total = Instant::now();

        let t = Instant::now();
        let emb = self.embed(text)?;
        let embed_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        let neighbors = self.nearest(agent_id, &emb, self.gate.neighbor_k).await?;
        let nearest_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        let novelty = compression_novelty(text, &neighbors);
        let salience = rule_salience(text);
        let correction = is_correction(text);
        let signals_ms = t.elapsed().as_secs_f64() * 1e3;

        let (admitted, reason) = if correction {
            (true, "correction-bypass")
        } else if salience < self.gate.salience_floor {
            (false, "below salience floor")
        } else if novelty >= self.gate.novelty_threshold {
            (true, "novel enough")
        } else {
            (false, "redundant (low novelty)")
        };

        let mut id = None;
        let mut insert_ms = 0.0;
        if admitted {
            let t = Instant::now();
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as i64;
            let v = pgvector::Vector::from(emb);
            let new_id: i64 = sqlx::query_scalar(
                "INSERT INTO messages (agent_id, scope, role, text, ts, novelty, salience, embedding) \
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id",
            )
            .bind(agent_id)
            .bind(scope)
            .bind(role)
            .bind(text)
            .bind(ts)
            .bind(novelty as f32)
            .bind(salience as f32)
            .bind(&v)
            .fetch_one(&self.pool)
            .await?;
            id = Some(new_id);
            insert_ms = t.elapsed().as_secs_f64() * 1e3;
        }

        Ok(Decision {
            admitted,
            id,
            novelty,
            salience,
            correction,
            reason,
            timings: Timings {
                embed_ms,
                nearest_ms,
                signals_ms,
                insert_ms,
                total_ms: t_total.elapsed().as_secs_f64() * 1e3,
            },
        })
    }

    pub async fn count(&self) -> Result<i64> {
        Ok(sqlx::query_scalar("SELECT COUNT(*) FROM messages")
            .fetch_one(&self.pool)
            .await?)
    }

    /// Drop and recreate the messages table + indexes with a new vector dimension.
    /// Used by benchmarks that use a different embedding model than the default.
    ///
    /// Also truncates `consolidations`: message ids restart at 1 on every
    /// reset, so stale consolidation rows from a prior run would otherwise
    /// silently attach to the wrong (newly inserted) messages and make
    /// `pending_consolidation` skip them as "already done".
    pub async fn reset_for_dim(&self, dim: usize) -> Result<()> {
        sqlx::query("DROP TABLE IF EXISTS messages").execute(&self.pool).await?;
        sqlx::query(AssertSqlSafe(format!(
            "CREATE TABLE messages (
                id        BIGSERIAL PRIMARY KEY,
                agent_id  TEXT NOT NULL,
                scope     TEXT NOT NULL,
                role      TEXT NOT NULL,
                text      TEXT NOT NULL,
                ts        BIGINT NOT NULL,
                novelty   REAL,
                salience  REAL,
                embedding VECTOR({dim})
            )"
        )))
        .execute(&self.pool)
        .await?;
        sqlx::query(AssertSqlSafe(
            "CREATE INDEX messages_embedding_hnsw ON messages \
             USING hnsw (embedding vector_cosine_ops)".to_string(),
        ))
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "CREATE INDEX messages_bm25 ON messages \
             USING bm25 (id, text) WITH (key_field = 'id')",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query("TRUNCATE consolidations").execute(&self.pool).await?;
        Ok(())
    }

    /// Remove all messages for a clean demo/test run.
    pub async fn truncate(&self) -> Result<()> {
        sqlx::query("TRUNCATE messages RESTART IDENTITY")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    // ── M5 consolidation DB ops ──────────────────────────────────────────────

    /// Count of messages not yet consolidated by `model`. Used by callers to
    /// detect "fully drained" vs. "stuck on persistently-failing messages"
    /// without relying on a per-tick attempted/succeeded count.
    pub async fn pending_consolidation_count(&self, model: &str) -> Result<i64> {
        let n: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM messages \
             WHERE id NOT IN (SELECT message_id FROM consolidations WHERE model = $1)",
        )
        .bind(model)
        .fetch_one(&self.pool)
        .await?;
        Ok(n)
    }

    /// Messages not yet consolidated by `model`, grouped by agent (so the
    /// caller can cluster contiguous rows per-agent) then newest first within
    /// each agent.
    pub async fn pending_consolidation(
        &self,
        model: &str,
        limit: i64,
    ) -> Result<Vec<(i64, String, String)>> {
        let rows: Vec<(i64, String, String)> = sqlx::query_as(
            "SELECT id, agent_id, text FROM messages \
             WHERE id NOT IN \
               (SELECT message_id FROM consolidations WHERE model = $1) \
             ORDER BY agent_id, ts DESC LIMIT $2",
        )
        .bind(model)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Upsert a consolidation result. Fields are passed individually to avoid
    /// a crate circular dependency between `store` and `consolidation`.
    pub async fn store_consolidation(
        &self,
        message_id: i64,
        topic_path: &str,
        title: &str,
        body: &str,
        foresight_json: &str,
        model: &str,
    ) -> Result<i64> {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let id: i64 = sqlx::query_scalar(
            "INSERT INTO consolidations \
               (message_id, topic_path, title, body, foresight, model, ts) \
             VALUES ($1, $2, $3, $4, $5, $6, $7) \
             ON CONFLICT (message_id, model) DO UPDATE \
               SET topic_path=$2, title=$3, body=$4, foresight=$5, ts=$7 \
             RETURNING id",
        )
        .bind(message_id)
        .bind(topic_path)
        .bind(title)
        .bind(body)
        .bind(foresight_json)
        .bind(model)
        .bind(ts)
        .fetch_one(&self.pool)
        .await?;
        Ok(id)
    }

    /// Fetch the consolidation for a specific message + model, if it exists.
    pub async fn get_consolidation(
        &self,
        message_id: i64,
        model: &str,
    ) -> Result<Option<(String, String, String, String)>> {
        let row: Option<(String, String, String, String)> = sqlx::query_as(
            "SELECT topic_path, title, body, foresight \
             FROM consolidations WHERE message_id = $1 AND model = $2",
        )
        .bind(message_id)
        .bind(model)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row)
    }

    // ── High-level memory API (used by CLI + MCP server in M4) ─────────────

    /// Elevate a private memory to shared scope (visible to all agents).
    /// Returns `false` if the id doesn't exist or doesn't belong to `agent_id`.
    pub async fn promote(&self, agent_id: &str, id: i64) -> Result<bool> {
        let r = sqlx::query(
            "UPDATE messages SET scope = 'shared' WHERE id = $1 AND agent_id = $2",
        )
        .bind(id)
        .bind(agent_id)
        .execute(&self.pool)
        .await?;
        Ok(r.rows_affected() > 0)
    }

    /// Delete a memory. Returns `false` if the id doesn't exist or doesn't
    /// belong to `agent_id`.
    pub async fn forget(&self, agent_id: &str, id: i64) -> Result<bool> {
        let r = sqlx::query("DELETE FROM messages WHERE id = $1 AND agent_id = $2")
            .bind(id)
            .bind(agent_id)
            .execute(&self.pool)
            .await?;
        Ok(r.rows_affected() > 0)
    }

    /// Memory statistics for one agent.
    pub async fn profile(&self, agent_id: &str) -> Result<Profile> {
        let total: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM messages WHERE agent_id = $1")
                .bind(agent_id)
                .fetch_one(&self.pool)
                .await?;
        let shared: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM messages WHERE agent_id = $1 AND scope = 'shared'",
        )
        .bind(agent_id)
        .fetch_one(&self.pool)
        .await?;
        let avg_novelty: Option<f64> =
            sqlx::query_scalar("SELECT AVG(novelty) FROM messages WHERE agent_id = $1")
                .bind(agent_id)
                .fetch_one(&self.pool)
                .await?;
        let avg_salience: Option<f64> =
            sqlx::query_scalar("SELECT AVG(salience) FROM messages WHERE agent_id = $1")
                .bind(agent_id)
                .fetch_one(&self.pool)
                .await?;
        let oldest_ts: Option<i64> =
            sqlx::query_scalar("SELECT MIN(ts) FROM messages WHERE agent_id = $1")
                .bind(agent_id)
                .fetch_one(&self.pool)
                .await?;
        let newest_ts: Option<i64> =
            sqlx::query_scalar("SELECT MAX(ts) FROM messages WHERE agent_id = $1")
                .bind(agent_id)
                .fetch_one(&self.pool)
                .await?;
        Ok(Profile {
            agent_id: agent_id.to_string(),
            total,
            private: total - shared,
            shared,
            avg_novelty: avg_novelty.unwrap_or(0.0),
            avg_salience: avg_salience.unwrap_or(0.0),
            oldest_ts,
            newest_ts,
        })
    }

    /// Most recent memories for an agent, newest first.
    pub async fn list(&self, agent_id: &str, limit: i64) -> Result<Vec<MemoryRow>> {
        let rows: Vec<(i64, String, String, String, i64, Option<f32>, Option<f32>)> =
            sqlx::query_as(
                "SELECT id, scope, role, text, ts, novelty, salience \
                 FROM messages WHERE agent_id = $1 ORDER BY ts DESC LIMIT $2",
            )
            .bind(agent_id)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;
        Ok(rows
            .into_iter()
            .map(|(id, scope, role, text, ts, novelty, salience)| MemoryRow {
                id,
                scope,
                role,
                text,
                ts,
                novelty,
                salience,
            })
            .collect())
    }

    /// All messages for one agent, in insertion order. `id` (BIGSERIAL) is
    /// used rather than `ts` — a tight insert loop can produce duplicate
    /// millisecond timestamps, but ids are strictly monotonic. Used by eval
    /// harnesses to recover a turn-index -> message-id mapping after an
    /// interrupted run, without re-inserting (and re-paying for embeddings).
    pub async fn messages_for_agent(&self, agent_id: &str) -> Result<Vec<(i64, String)>> {
        let rows: Vec<(i64, String)> = sqlx::query_as(
            "SELECT id, text FROM messages WHERE agent_id = $1 ORDER BY id ASC",
        )
        .bind(agent_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }
}
