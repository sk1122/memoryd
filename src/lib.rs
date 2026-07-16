// Re-export from `signals` crate. The implementation lives there so that
// `crates/store` can depend on it without creating a circular dependency
// with the root `memoryd` package (which depends on `store` for the CLI).
pub use signals::{compression_novelty, gzip_len, is_correction, rule_salience};
