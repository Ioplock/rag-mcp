//! Local sentence embeddings via candle (CPU, no ONNX runtime).
//!
//! Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
//! (multilingual incl. Russian, 384 dims). Weights live in RAG_MODEL_DIR
//! (default `~/.cache/rag-mcp/model`) — download once with:
//!   mkdir -p ~/.cache/rag-mcp/model && cd ~/.cache/rag-mcp/model
//!   for f in config.json tokenizer.json model.safetensors; do
//!     curl -L -o $f https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/$f
//!   done

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::Tokenizer;

pub const DIM: usize = 384;
const BATCH: usize = 32;

pub fn model_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("RAG_MODEL_DIR") {
        return d.into();
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home).join(".cache/rag-mcp/model")
}

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub fn load(cfg: &crate::config::Config) -> Result<Self> {
        let dir = std::path::PathBuf::from(cfg.model_dir.clone());
        let get = |f: &str| {
            let p = dir.join(f);
            if !p.exists() {
                anyhow::bail!(
                    "model file missing: {}\nDownload with:\n  mkdir -p {}\n  for f in config.json tokenizer.json model.safetensors; do curl -L -o {}/$f https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/$f; done\n(or set RAG_MODEL_DIR)",
                    p.display(),
                    dir.display(),
                    dir.display()
                );
            }
            Ok(p)
        };
        let config_path = get("config.json")?;
        let tokenizer_path = get("tokenizer.json")?;
        let weights_path = get("model.safetensors")?;
        let config: Config =
            serde_json::from_slice(&std::fs::read(config_path).context("read config")?)
                .context("parse bert config")?;
        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .context("load weights")?
        };
        let model = BertModel::load(vb, &config).context("build bert model")?;
        Ok(Self { model, tokenizer, device })
    }

    /// Mean-pool over non-padding tokens, then L2-normalize.
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for batch in texts.chunks(BATCH) {
            let enc = self
                .tokenizer
                .encode_batch(batch.to_vec(), true)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let max_len = enc.iter().map(|e| e.len()).max().unwrap_or(0).max(1);
            let mut ids: Vec<u32> = Vec::with_capacity(batch.len() * max_len);
            let mut mask: Vec<u32> = Vec::with_capacity(batch.len() * max_len);
            for e in &enc {
                let mut t = e.get_ids().to_vec();
                let mut m = e.get_attention_mask().to_vec();
                t.resize(max_len, 0);
                m.resize(max_len, 0);
                ids.extend_from_slice(&t);
                mask.extend_from_slice(&m);
            }
            let ids = Tensor::new(ids, &self.device)?.reshape((batch.len(), max_len))?;
            let mask = Tensor::new(mask, &self.device)?.reshape((batch.len(), max_len))?;
            let emb = self.model.forward(&ids, &mask, None).context("bert forward")?;
            // masked mean pooling
            let mask_f = mask.to_dtype(DType::F32)?.unsqueeze(2)?; // [b, n, 1]
            let summed = emb.broadcast_mul(&mask_f)?.sum(1)?; // [b, h]
            let counts = mask_f.sum_keepdim((1, 2))?.squeeze(2)?.squeeze(1)?; // [b]
            let counts = counts.clamp(1e-9, f64::MAX)?.unsqueeze(1)?; // [b, 1]
            let mean = summed.broadcast_div(&counts)?; // [b, h]
            let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-12, f64::MAX)?;
            let unit = mean.broadcast_div(&norm)?;
            let flat: Vec<f32> = unit.flatten_all()?.to_vec1()?;
            let h = flat.len() / batch.len();
            for (i, row) in flat.chunks(h).enumerate() {
                let _ = i;
                out.push(row.to_vec());
            }
        }
        Ok(out)
    }

    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed(&[text.to_string()])?.pop().unwrap_or_default())
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f32;
    let (mut na, mut nb) = (0f32, 0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

pub fn to_blob(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

pub fn from_blob(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blob_roundtrip() {
        let v = vec![0.5f32, -1.25, 3.0];
        assert_eq!(from_blob(&to_blob(&v)), v);
    }
    #[test]
    fn cosine_self_is_one() {
        let v = vec![0.2f32, 0.8, -0.1];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-5);
    }
}
