//! Deterministic weight initialisation utilities.
//!
//! The original C implementation that inspired this project exposed a manual
//! initialisation routine.  In Rust we model that behaviour through the
//! [`WeightInitializer`] type which feeds pseudo random numbers into the model
//! layers.  Keeping the generator extremely small makes it simple to understand
//! the values that end up in the network which is perfect for experimentation
//! and educational purposes.

use crate::layers::{FeedForward, LayerNorm, Linear, MultiHeadAttention};

pub struct WeightInitializer {
    state: u64,
}

impl WeightInitializer {
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x_4d595df4_d0f33173
        } else {
            seed
        };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.state = x;
        x
    }

    pub fn next_f32(&mut self) -> f32 {
        let value = self.next_u64() as f64 / u64::MAX as f64;
        (value * 2.0 - 1.0) as f32
    }

    pub fn vector(&mut self, len: usize, scale: f32) -> Vec<f32> {
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            data.push(self.next_f32() * scale);
        }
        data
    }

    pub fn embeddings(&mut self, rows: usize, cols: usize) -> Vec<f32> {
        self.vector(rows * cols, 0.02)
    }

    pub fn linear(&mut self, in_features: usize, out_features: usize) -> Linear {
        let scale = 1.0 / (in_features as f32).sqrt();
        let weight = self.vector(in_features * out_features, scale);
        let bias = vec![0.0; out_features];
        Linear::new(in_features, out_features, weight, bias)
    }

    pub fn layer_norm(&mut self, hidden_size: usize) -> LayerNorm {
        let mut gamma = vec![1.0; hidden_size];
        for value in gamma.iter_mut() {
            *value += self.next_f32() * 0.01;
        }
        let beta = vec![0.0; hidden_size];
        LayerNorm::new(hidden_size, gamma, beta, 1e-5)
    }

    pub fn attention(&mut self, hidden_size: usize, n_heads: usize) -> MultiHeadAttention {
        let w_q = self.linear(hidden_size, hidden_size);
        let w_k = self.linear(hidden_size, hidden_size);
        let w_v = self.linear(hidden_size, hidden_size);
        let w_o = self.linear(hidden_size, hidden_size);
        MultiHeadAttention::new(hidden_size, n_heads, w_q, w_k, w_v, w_o)
    }

    pub fn feed_forward(&mut self, hidden_size: usize, intermediate: usize) -> FeedForward {
        let proj_in = self.linear(hidden_size, intermediate);
        let proj_out = self.linear(intermediate, hidden_size);
        FeedForward::new(proj_in, proj_out)
    }
}
