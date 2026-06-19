-- Ensure the extensions exist on first container init. The actual schema
-- (tables, indexes) is owned and migrated by the Rust store, not here.
CREATE EXTENSION IF NOT EXISTS vector;     -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_search;  -- ParadeDB BM25
