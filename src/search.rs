//! Hybrid retrieval: 0.5 * norm(BM25) + 0.5 * cosine.

use anyhow::Result;
use rusqlite::Connection;
use std::collections::HashMap;

pub struct Hit {
    pub doc: String,
    pub chunk_idx: i64,
    pub text: String,
    pub score: f32,
    pub bm25: Option<f64>,
    pub cosine: f32,
}

fn minmax(vals: &HashMap<i64, f64>, higher_better: bool) -> HashMap<i64, f32> {
    let mut out = HashMap::new();
    if vals.is_empty() {
        return out;
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for v in vals.values() {
        min = min.min(*v);
        max = max.max(*v);
    }
    let span = (max - min).max(1e-12);
    for (k, v) in vals {
        let n = (v - min) / span;
        out.insert(*k, if higher_better { n as f32 } else { (1.0 - n) as f32 });
    }
    out
}

pub fn hybrid_search(
    conn: &Connection,
    embedder: &crate::embed::Embedder,
    query: &str,
    top_k: usize,
) -> Result<Vec<Hit>> {
    const CANDIDATES: i64 = 120;
    let top_k = top_k.clamp(1, 50);

    let fts = crate::store::fts_search(conn, query, CANDIDATES)?;
    let fts_map: HashMap<i64, f64> = fts.into_iter().collect();

    let q = embedder.embed_one(query)?;
    let all = crate::store::all_chunk_embeddings(conn)?;
    let mut cos_map: HashMap<i64, f64> = HashMap::with_capacity(all.len());
    for (id, emb) in &all {
        cos_map.insert(*id, crate::embed::cosine(&q, emb) as f64);
    }
    // Keep only top-CANDIDATES by cosine to bound work.
    let mut cos_top: Vec<(i64, f64)> = cos_map.into_iter().collect();
    cos_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    cos_top.truncate(CANDIDATES as usize);
    let cos_map: HashMap<i64, f64> = cos_top.into_iter().collect();

    let fts_n = minmax(&fts_map, false); // bm25: lower is better
    let cos_n = minmax(&cos_map, true);

    let mut ids: Vec<i64> = fts_map.keys().copied().collect();
    for id in cos_map.keys() {
        if !fts_map.contains_key(id) {
            ids.push(*id);
        }
    }
    let mut scored: Vec<(i64, f32)> = ids
        .into_iter()
        .map(|id| {
            let s = 0.5 * fts_n.get(&id).copied().unwrap_or(0.0)
                + 0.5 * cos_n.get(&id).copied().unwrap_or(0.0);
            (id, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);

    let mut hits = Vec::with_capacity(scored.len());
    for (id, s) in scored {
        if let Some(row) = crate::store::chunk_by_id(conn, id)? {
            hits.push(Hit {
                doc: row.doc,
                chunk_idx: row.idx,
                text: row.text,
                score: s,
                bm25: fts_map.get(&id).copied(),
                cosine: cos_map.get(&id).copied().unwrap_or(0.0) as f32,
            });
        }
    }
    Ok(hits)
}
