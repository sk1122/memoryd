//! M0 Spike 2: HuggingFace small embedding model in Rust.
//!
//! Model: BAAI/bge-small-en-v1.5 (384-dim, ~33M params) via fastembed (ort
//! backend). Validates not just "it returns numbers" but that the vectors are
//! semantically meaningful: two auth-related sentences should be much closer
//! than either is to an unrelated sentence about a dog.

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn main() -> anyhow::Result<()> {
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(true),
    )?;

    let docs = vec![
        "I rotated the JWT signing secret to v9",          // 0
        "We standardized on jsonwebtoken version 9 for auth", // 1 (related to 0)
        "The dog needs a vet appointment next Tuesday",     // 2 (unrelated)
    ];

    let emb = model.embed(docs.clone(), None)?;

    println!("embedded {} docs, dim = {}", emb.len(), emb[0].len());
    println!();
    println!("cosine(auth0, auth1) = {:.4}  <- should be HIGH", cosine(&emb[0], &emb[1]));
    println!("cosine(auth0, dog)   = {:.4}  <- should be LOW", cosine(&emb[0], &emb[2]));
    println!("cosine(auth1, dog)   = {:.4}  <- should be LOW", cosine(&emb[1], &emb[2]));

    let related = cosine(&emb[0], &emb[1]);
    let unrelated = cosine(&emb[0], &emb[2]);
    assert!(emb[0].len() == 384, "expected 384-dim, got {}", emb[0].len());
    assert!(
        related > unrelated + 0.1,
        "embeddings not semantically meaningful: related={related:.3} unrelated={unrelated:.3}"
    );
    println!("\nSPIKE PASS: 384-dim, related pair clearly closer than unrelated.");
    Ok(())
}
