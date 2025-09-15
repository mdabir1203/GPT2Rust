//! High level GPT-2 style network implementation.
//!
//! The goal of this module is not to be a drop-in replacement for the original
//! OpenAI model but to provide an approachable pure Rust implementation that can
//! be inspected and experimented with.

use crate::layers::{FeedForward, LayerNorm, Linear, MultiHeadAttention};
use crate::loader::WeightInitializer;

#[derive(Debug, Clone)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
}

impl Default for Gpt2Config {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            n_positions: 64,
            n_ctx: 64,
            n_embd: 128,
            n_layer: 4,
            n_head: 4,
        }
    }
}

impl Gpt2Config {
    pub fn small() -> Self {
        Self::default()
    }
}

struct TransformerBlock {
    attn_norm: LayerNorm,
    attention: MultiHeadAttention,
    mlp_norm: LayerNorm,
    mlp: FeedForward,
}

impl TransformerBlock {
    fn new(config: &Gpt2Config, init: &mut WeightInitializer) -> Self {
        let attn_norm = init.layer_norm(config.n_embd);
        let attention = init.attention(config.n_embd, config.n_head);
        let mlp_norm = init.layer_norm(config.n_embd);
        let mlp = init.feed_forward(config.n_embd, config.n_embd * 4);
        Self {
            attn_norm,
            attention,
            mlp_norm,
            mlp,
        }
    }

    fn forward(&self, hidden_states: &mut [f32], seq_len: usize) {
        let hidden_size = self.attention.hidden_size();
        let mut normed = hidden_states.to_vec();
        self.attn_norm.forward(&mut normed, seq_len, hidden_size);
        let attn_out = self.attention.forward(&normed, seq_len);
        for (value, residual) in hidden_states.iter_mut().zip(attn_out.iter()) {
            *value += residual;
        }

        let mut mlp_input = hidden_states.to_vec();
        self.mlp_norm.forward(&mut mlp_input, seq_len, hidden_size);
        let mlp_out = self.mlp.forward(&mlp_input, seq_len);
        for (value, residual) in hidden_states.iter_mut().zip(mlp_out.iter()) {
            *value += residual;
        }
    }
}

pub struct Gpt2Model {
    pub config: Gpt2Config,
    token_embedding: Vec<f32>,
    position_embedding: Vec<f32>,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    lm_head: Linear,
}

impl Gpt2Model {
    pub fn initialise(config: Gpt2Config, seed: u64) -> Self {
        let mut init = WeightInitializer::new(seed);
        let token_embedding = init.embeddings(config.vocab_size, config.n_embd);
        let position_embedding = init.embeddings(config.n_positions, config.n_embd);
        let mut blocks = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            blocks.push(TransformerBlock::new(&config, &mut init));
        }
        let final_norm = init.layer_norm(config.n_embd);
        let lm_head = init.linear(config.n_embd, config.vocab_size);
        Self {
            config,
            token_embedding,
            position_embedding,
            blocks,
            final_norm,
            lm_head,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let vocab = self.config.vocab_size;
        text.bytes().map(|b| (b as usize) % vocab).collect()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&id| {
                let byte = (id % 256) as u8;
                byte as char
            })
            .collect()
    }

    pub fn forward(&self, tokens: &[usize]) -> Vec<f32> {
        if tokens.is_empty() {
            return Vec::new();
        }
        let seq_len = tokens.len().min(self.config.n_ctx);
        let hidden_size = self.config.n_embd;
        let mut hidden_states = vec![0.0; seq_len * hidden_size];
        for (position, &token_id) in tokens.iter().take(seq_len).enumerate() {
            let token_index = token_id % self.config.vocab_size;
            let pos_index = position % self.config.n_positions;
            let embed_offset = token_index * hidden_size;
            let pos_offset = pos_index * hidden_size;
            let hidden_offset = position * hidden_size;
            for dim in 0..hidden_size {
                hidden_states[hidden_offset + dim] = self.token_embedding[embed_offset + dim]
                    + self.position_embedding[pos_offset + dim];
            }
        }

        for block in &self.blocks {
            block.forward(&mut hidden_states, seq_len);
        }

        let mut normed = hidden_states.clone();
        self.final_norm
            .forward(&mut normed, seq_len, self.config.n_embd);
        self.lm_head.forward(&normed, seq_len)
    }

    pub fn generate(&self, prompt: &str, steps: usize) -> String {
        let mut tokens = self.encode(prompt);
        if tokens.is_empty() {
            tokens.push(b' ' as usize);
        }
        for _ in 0..steps {
            let context_len = tokens.len().min(self.config.n_ctx);
            let start = tokens.len().saturating_sub(context_len);
            let context = &tokens[start..];
            let logits = self.forward(context);
            if logits.is_empty() {
                break;
            }
            let vocab = self.config.vocab_size;
            let last = &logits[(context_len - 1) * vocab..context_len * vocab];
            let mut best_index = 0;
            let mut best_score = f32::NEG_INFINITY;
            for (idx, &score) in last.iter().enumerate() {
                if score > best_score {
                    best_score = score;
                    best_index = idx;
                }
            }
            tokens.push(best_index);
        }
        self.decode(&tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let model = Gpt2Model::initialise(Gpt2Config::default(), 42);
        let text = "Hello";
        let tokens = model.encode(text);
        let decoded = model.decode(&tokens);
        assert_eq!(decoded, text);
    }

    #[test]
    fn forward_respects_context_limit() {
        let mut config = Gpt2Config::default();
        config.n_ctx = 4;
        let model = Gpt2Model::initialise(config, 123);
        let tokens = vec![1, 2, 3, 4, 5, 6];
        let logits = model.forward(&tokens);
        assert_eq!(logits.len(), 4 * model.config.vocab_size);
    }
}
