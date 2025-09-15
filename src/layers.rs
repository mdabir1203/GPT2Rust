//! Core building blocks used by the GPT-2 style model.
//!
//! The implementation favours clarity over raw performance.  The tensor
//! operations are written in straight-forward Rust without any unsafe math or
//! external dependencies which makes the code easy to inspect and extend.

use std::f32;

fn softmax(values: &mut [f32]) {
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum == 0.0 {
        return;
    }
    let inv_sum = 1.0 / sum;
    for value in values.iter_mut() {
        *value *= inv_sum;
    }
}

#[derive(Clone)]
pub struct Linear {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, weight: Vec<f32>, bias: Vec<f32>) -> Self {
        assert_eq!(weight.len(), in_features * out_features);
        assert_eq!(bias.len(), out_features);
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    pub fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        assert_eq!(input.len(), seq_len * self.in_features);
        let mut output = vec![0.0; seq_len * self.out_features];
        for token in 0..seq_len {
            let input_offset = token * self.in_features;
            let output_offset = token * self.out_features;
            let input_slice = &input[input_offset..input_offset + self.in_features];
            let output_slice = &mut output[output_offset..output_offset + self.out_features];
            for out_idx in 0..self.out_features {
                let mut acc = self.bias[out_idx];
                let weight_offset = out_idx * self.in_features;
                for in_idx in 0..self.in_features {
                    acc += self.weight[weight_offset + in_idx] * input_slice[in_idx];
                }
                output_slice[out_idx] = acc;
            }
        }
        output
    }
}

#[derive(Clone)]
pub struct LayerNorm {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub epsilon: f32,
}

impl LayerNorm {
    pub fn new(hidden_size: usize, gamma: Vec<f32>, beta: Vec<f32>, epsilon: f32) -> Self {
        assert_eq!(gamma.len(), hidden_size);
        assert_eq!(beta.len(), hidden_size);
        Self {
            gamma,
            beta,
            epsilon,
        }
    }

    pub fn forward(&self, values: &mut [f32], seq_len: usize, hidden_size: usize) {
        assert_eq!(values.len(), seq_len * hidden_size);
        for token in 0..seq_len {
            let offset = token * hidden_size;
            let slice = &mut values[offset..offset + hidden_size];
            let mut mean = 0.0;
            for &value in slice.iter() {
                mean += value;
            }
            mean /= hidden_size as f32;

            let mut variance = 0.0;
            for &value in slice.iter() {
                let diff = value - mean;
                variance += diff * diff;
            }
            variance /= hidden_size as f32;
            let denom = (variance + self.epsilon).sqrt();

            for (idx, value) in slice.iter_mut().enumerate() {
                let normalized = (*value - mean) / denom;
                *value = normalized * self.gamma[idx] + self.beta[idx];
            }
        }
    }
}

#[derive(Clone)]
pub struct MultiHeadAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    n_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    scale: f32,
}

impl MultiHeadAttention {
    pub fn new(
        hidden_size: usize,
        n_heads: usize,
        w_q: Linear,
        w_k: Linear,
        w_v: Linear,
        w_o: Linear,
    ) -> Self {
        assert_eq!(
            hidden_size % n_heads,
            0,
            "hidden size must be divisible by number of heads"
        );
        let head_dim = hidden_size / n_heads;
        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            n_heads,
            head_dim,
            hidden_size,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn forward(&self, hidden_states: &[f32], seq_len: usize) -> Vec<f32> {
        assert_eq!(hidden_states.len(), seq_len * self.hidden_size);
        if seq_len == 0 {
            return Vec::new();
        }
        let query = self.w_q.forward(hidden_states, seq_len);
        let key = self.w_k.forward(hidden_states, seq_len);
        let value = self.w_v.forward(hidden_states, seq_len);
        let mut combined = vec![0.0; seq_len * self.hidden_size];
        let mut scores = vec![0.0; seq_len];

        for token in 0..seq_len {
            for head in 0..self.n_heads {
                let head_offset = head * self.head_dim;
                let query_offset = token * self.hidden_size + head_offset;
                for past in 0..=token {
                    let key_offset = past * self.hidden_size + head_offset;
                    let mut dot = 0.0;
                    for dim in 0..self.head_dim {
                        dot += query[query_offset + dim] * key[key_offset + dim];
                    }
                    scores[past] = dot * self.scale;
                }
                softmax(&mut scores[..token + 1]);
                for dim in 0..self.head_dim {
                    let mut value_acc = 0.0;
                    for past in 0..=token {
                        let val_offset = past * self.hidden_size + head_offset;
                        value_acc += scores[past] * value[val_offset + dim];
                    }
                    combined[query_offset + dim] = value_acc;
                }
                for past in 0..=token {
                    scores[past] = 0.0;
                }
            }
        }

        self.w_o.forward(&combined, seq_len)
    }
}

#[derive(Clone)]
pub struct FeedForward {
    proj_in: Linear,
    proj_out: Linear,
}

impl FeedForward {
    pub fn new(proj_in: Linear, proj_out: Linear) -> Self {
        Self { proj_in, proj_out }
    }

    pub fn forward(&self, hidden_states: &[f32], seq_len: usize) -> Vec<f32> {
        let intermediate = self.proj_in.forward(hidden_states, seq_len);
        let activated: Vec<f32> = intermediate.into_iter().map(gelu).collect();
        self.proj_out.forward(&activated, seq_len)
    }
}

fn gelu(x: f32) -> f32 {
    const COEFF: f32 = std::f32::consts::FRAC_2_SQRT_PI;
    0.5 * x * (1.0 + (COEFF * (x + 0.044_715 * x.powi(3))).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_normalizes() {
        let mut values = vec![1.0, 2.0, 3.0];
        softmax(&mut values);
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn linear_forward_shape() {
        let layer = Linear::new(2, 3, vec![1.0; 6], vec![0.0; 3]);
        let out = layer.forward(&[1.0, 2.0, -1.0, 0.5], 2);
        assert_eq!(out.len(), 6);
    }
}
