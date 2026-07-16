# memoryd

A local-first, sovereign memory substrate for AI agents. Stores raw conversation turns, gates
redundant writes with a compression-based novelty signal, and retrieves relevant memories through
a hybrid BM25 + dense-vector + cross-encoder pipeline — all without an LLM on the hot path.

## Papers

Two papers drive the design:

**TrueMemory** — *"True Memory: A Practical Long-Term Memory for AI Agents"* (arXiv:2605.04897)  
Core principle: *store-raw, interpret-late*. Keep every turn verbatim; use cheap signals (gzip
compression distance, rule-based salience) to gate redundancy; defer LLM interpretation to a
background consolidation worker. Benchmarked on LoCoMo with 89.6% QA accuracy at the Edge tier.

**EverMemOS** — *"EverMemOS: A Memory Operating System for AI Agents"* (arXiv:2601.02163)  
Provides the higher-level architecture: MemCells (individual memory units with validity intervals)
and MemScenes (topic-grouped clusters). Introduces foresight extraction — LLM-predicted future
relevance attached at write time — and semantic consolidation for merging related memories over time.
memoryd's `ConsolidationModel` trait (M5) is the interface point for this.

## Architecture

```
memoryd (lib)          novelty + salience + correction signals
crates/store           storage core, retrieval pipeline, ingestion gate
crates/spikes          one-shot integration probes (M0)
src/bin/memplay.rs     interactive developer harness (REPL) for the substrate
src/bin/memchat.rs     memory-grounded chat TUI (ratatui + crossterm)
docker/                ParadeDB (Postgres 18 + pg_search + pgvector) container
data/                  locomo10.json benchmark + embedding caches
```

### Signal layer — `src/lib.rs`

Three LLM-free signals computed on every ingest:

| Signal | Implementation | Paper reference |
|--------|---------------|-----------------|
| **Compression novelty** | `(gzip(mem+fact) - gzip(mem)) / gzip(fact)` — NCD proxy | TrueMemory §3.1 |
| **Rule salience** | Length buckets + digit/keyword boosts | TrueMemory §3.2 |
| **Correction bypass** | Regex markers + "not X but Y" pattern | TrueMemory §3.3 |

Uses `flate2` with the `zlib-rs` backend (pure-Rust port of zlib) for byte-identical output to
CPython's `zlib` module. Validated on the LoCoMo dataset: AUC 0.769 (TrueMemory reports 0.788).

### Storage + gate — `crates/store/src/lib.rs`

**Schema** — single `messages` table in ParadeDB:
- `VECTOR(N)` column indexed with HNSW (`vector_cosine_ops`) via pgvector
- BM25 index via `pg_search` extension (`USING bm25`) — real BM25, not Postgres `ts_rank`

**Ingestion gate** (TrueMemory §4): embed → fetch k nearest neighbors → compute novelty + salience
→ admit/drop. Corrections always bypass. Admitted rows are inserted with their embedding.

**Retrieval pipeline** (TrueMemory §5 + standard IR):
1. BM25 lexical search — `paradedb.match()` via `@@@` operator
2. Dense vector search — cosine ANN via pgvector HNSW
3. Reciprocal Rank Fusion (K=60) — merges the two ranked lists
4. Cross-encoder rerank — BGE-reranker-base via fastembed (ONNX, local)

```
query ──► BM25 (top-100) ─┐
      └─► vector (top-100) ┴─► RRF ──► top-2k candidates ──► reranker ──► top-k hits
```

**Multi-agent isolation**: every row carries `agent_id` and `scope` (`private` | `shared`).
Retrieval scopes to `agent_id = :me OR scope = 'shared'`.

### Consolidation stub — `crates/spikes/src/bin/spike_llm.rs`

Validates the `ConsolidationModel` trait against gpt-5-mini (hosted). This is the M5 interface
for EverMemOS-style background enrichment: structured output with `topic_path`, `title`, `body`,
and `foresight[]` fields (EverMemOS §3.2 — foresight extraction with validity intervals).
Production target: Gemma-3-4B local via `ort`.

## Build sequence

Milestones are validation-gated; each must pass before the next is started.

| Milestone | Description | Gate |
|-----------|-------------|------|
| **M1** | gzip novelty signal + LoCoMo AUC harness | AUC > 0.75 on locomo10 |
| **M0** | Integration spikes: ParadeDB, fastembed, gpt-5-mini consolidation | All three spikes pass |
| **M2** | Storage core + ingestion gate end-to-end | ingest_demo assertions pass |
| **M3** | Retrieval pipeline + LoCoMo recall benchmark | recall@100 > 90% (vector path) |
| M4 | MCP server (`rmcp`) — remember/recall/promote/forget/profile tools | scope isolation test |
| M5 | Consolidation worker — Gemma-3-4B local, extractive fallback | three-way eval (none/lazy/eager) |
| M6 | FUSE mount (`fuser`) — computed filesystem over the store | mount + read test |
| M7 | Agent runtime + scheduling | end-to-end agent loop |

M1–M3 are complete.

## LoCoMo benchmark results

Dataset: 10 conversations, 1,531 QA pairs (categories 1–4; adversarial excluded per TrueMemory
protocol). Embedding: Google `text-embedding-004` via Vertex AI. Retrieval: k=100.

### Retrieval recall@100

```
=== retrieval recall@100 [Google text-embedding-004 (768-dim)] ===
category        bm25   vector   fused    full    n
single-hop     52.4%   87.7%   85.0%   85.8%   281
multi-hop      83.7%   94.6%   94.7%   93.0%   320
temporal       45.6%   80.6%   75.9%   78.7%    89
open-domain    82.8%   96.3%   97.2%   96.8%   841
ALL            75.2%   93.5%   93.2%   92.9%  1531
```

Key finding: BM25 significantly underperforms dense retrieval on conversational turns. RRF fusion
adds marginal value over vector-only. With the candidate pool fixed to `2×k`, the `full` (reranked)
column now tracks the dense path closely instead of capping out.

### End-to-end QA accuracy

```
=== QA accuracy (gpt-5-mini answer + judge) ===
category       accuracy   n
single-hop        52.7%   281
multi-hop         67.2%   320
temporal          53.9%    89
open-domain       83.2%   841
ALL               72.6%  1531
```

**These two tables measure different things.** Retrieval recall (92.9%) asks "is the right memory in
the top-100?"; QA accuracy (72.6%) asks "did the reader produce the judged-correct answer given that
context?". The ~20-point gap is downstream of memoryd — the right memory is retrieved ~93% of the
time, but the answer-generation step loses ground on single-hop and temporal questions. The memory
substrate (gate + retrieval) is doing its job; the weakness is in the reader/judge, which is not
memoryd's contribution.

**QA setup caveat.** These numbers use a **single** `gpt-5-mini` model for both answering and judging
(one judge, no majority vote). TrueMemory's reported 89.6% uses a **3-judge majority vote** with a
stronger reader model — so 72.6% here is *not* apples-to-apples with their headline number. Switching
to a 3-judge panel and a higher-capability reader (e.g. a frontier model) is expected to raise QA
accuracy materially without touching the memory layer. The retrieval recall above is model-independent
and is the metric that reflects memoryd's actual work.

## Running

### Prerequisites

```bash
# Start ParadeDB (Postgres 18 + pg_search + pgvector)
docker compose -f docker/docker-compose.yml up -d

# Verify
psql postgres://memoryd:memoryd@localhost:5433/memoryd -c "SELECT version();"
```

### Run the M1 novelty AUC harness

```bash
cargo run --release --bin locomo_eval
# Expected: AUC ~0.769
```

### Run the M2 ingest demo

```bash
cargo run -p store --release --bin ingest_demo
```

### Run the interactive developer harness (memplay)

`memplay` is an interactive REPL for exploring the memory substrate live —
ingest through the gate, recall with the full retrieval pipeline, and flip
between raw / lazy-consolidated / eager-consolidated views of the same recall
to see what consolidation does. Requires the ParadeDB container.

```bash
cargo run --bin memplay -- [--agent <id>] [--fresh]
```

`--fresh` drops and recreates the `messages` table at 384-dim (bge-small) for a
clean slate — use it if a prior `locomo_qa` / `longmemeval_qa` run left the
table at a different vector width (the harness embeds with bge-small, 384-dim).

Inside the REPL:

```
<text>            ingest through the novelty + salience gate
?<query>          recall top-k (uses current :mode and :model)
:seed             load a scripted demo conversation (exercises every gate branch)
:list [n]         recent memories, newest first
:show <id>        raw memory + any stored consolidation
:consolidate [m]  run the consolidation worker once over pending memories
:profile          per-agent memory stats
:mode <m>         none | lazy | eager   (recall view)
:model <m>        extractive | openai   (consolidation model)
:scope <s>        private | shared      (ingest scope)
:role <r>         user | assistant | system
:agent <id>       switch active agent (multi-agent isolation demo)
:k <n>            recall top-k
:count            total rows in the store
:fresh            drop + recreate messages table at 384-dim (erases everything)
:reset            truncate the store (keeps vector dim)
:quit             exit
```

A typical showcase session:

```
:seed                       # load scripted conversation, watch the gate decide
?where does the user work   # recall raw hits (mode=none)
:mode eager
:consolidate                # run the extractive consolidator over pending
?where does the user work   # same recall, now showing consolidated bodies
:mode lazy
?tell me about Scout        # consolidate each hit on the fly
```

### Chat with it in the TUI (memchat)

`memchat` is a full-screen terminal UI for chatting with a memory-grounded
agent. Each message you send is ingested through the gate, used as a recall
query (BM25 + dense + RRF + cross-encoder), and answered by an LLM grounded in
the recalled memories; the assistant's reply is itself ingested so the
conversation has continuity — the agent remembers what it just told you.

```bash
# Echo mode — no API key needed; answers surface the recalled memories
cargo run --bin memchat -- [--agent <id>] [--fresh]

# Grounded LLM replies (gpt-5-mini)
OPENAI_API_KEY=sk-... cargo run --bin memchat -- [--agent <id>] [--fresh]
```

`--fresh` drops + recreates the `messages` table at 384-dim (bge-small) — use it
if a prior `locomo_qa` / `longmemeval_qa` run left the table at 768-dim.

Layout: chat transcript (left) + recalled-memory panel (right, shows the hits
grounding the latest reply with ids + scores) + status bar + input box.

```
<text>        send a message (ingested → recalled → answered)
:<command>    run a command
↑ / ↓         scroll transcript   PgUp/PgDn · Home/End
?             toggle help
Esc / Ctrl-C  quit
```

Commands: `:seed` `:agent <id>` `:scope private|shared` `:mode none|lazy|eager`
`:model extractive|openai` `:k <n>` `:consolidate [m]` `:profile` `:fresh y`
`:reset y` `:clear` `:help`.

A pty smoke-test harness lives at `scripts/smoke_memchat.py` (drives the TUI
through a pseudo-terminal: `:seed` → a chat message → Ctrl-C, and asserts a
clean exit).

### Run the M2 ingestion benchmark

```bash
cargo run -p store --release --bin bench_ingest
# Expected: ~28 msgs/sec, embed dominates at ~75% of wall time
```

### Run the M3 LoCoMo retrieval benchmark

```bash
# Retrieval recall only (no API key needed, uses cached embeddings on repeat runs)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/vertex-sa.json \
  cargo run -p store --release --bin locomo_qa -- --convs 10 --k 100

# Full QA accuracy (parallel OpenAI calls, ~5 min)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/vertex-sa.json \
OPENAI_API_KEY=sk-... \
  cargo run -p store --release --bin locomo_qa -- --convs 10 --k 100 --qa

# Quick smoke test (100 questions only)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/vertex-sa.json \
OPENAI_API_KEY=sk-... \
  cargo run -p store --release --bin locomo_qa -- --convs 10 --k 100 --qa --max-q 100

# Background run with log
nohup cargo run -p store --release --bin locomo_qa -- --convs 10 --k 100 --qa \
  > logs/locomo_qa.log 2>&1 &
tail -f logs/locomo_qa.log
```

### Embedding cache

On the first run, turn and question embeddings are saved to:
```
data/cache_turns_5882turns.bin     (~18 MB, 768-dim)
data/cache_questions_1531questions.bin  (~4.7 MB)
```
Subsequent runs load from cache and skip all API calls. Delete the files to force re-embedding.

## Configuration

`memoryd.toml` is loaded at startup (hot-reloadable in future):

```toml
[database]
url = "postgres://memoryd:memoryd@localhost:5433/memoryd"

[gate]
novelty_threshold = 0.30   # gzip NCD cutoff — below this = redundant
salience_floor = 0.10      # rule salience floor — below this = noise
neighbor_k = 10            # nearest neighbors used as gzip context

[embedding]
model = "bge-small-en-v1.5"
dim = 384
```

## Key dependencies

| Crate | Role |
|-------|------|
| `fastembed` | Local ONNX inference — bge-small-en-v1.5 (embed) + BGE-reranker-base (rerank) |
| `pgvector` | Rust bindings for pgvector `VECTOR` type in sqlx |
| `sqlx` 0.9 | Async Postgres driver (pinned to 0.9 for pgvector compatibility) |
| `flate2` (zlib-rs) | gzip novelty — pure-Rust zlib matching CPython byte-for-byte |
| `jsonwebtoken` | Service account JWT for Google Vertex AI auth |
| `reqwest` | Async HTTP — Vertex AI embedding API + OpenAI QA/judge calls |
| `ratatui` + `crossterm` | `memchat` TUI — chat with a memory-grounded agent |
