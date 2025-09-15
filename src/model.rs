//! High level GPT-2 style network implementation.
//!
//! The goal of this module is not to be a drop-in replacement for the original
//! OpenAI model but to provide an approachable pure Rust implementation that can
//! be inspected and experimented with.

use std::fs::File;
use std::io::Read;

pub const NUM_PARAMETER_TENSORS: usize = 16;
pub const NUM_ACTIVATION_TENSORS: usize = 23;

#[derive(Debug, Clone, Copy)]
pub struct Gpt2Config {
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub channels: usize,
}

pub struct ParameterTensors<'a> {
    pub wte: &'a mut [f32],
    pub wpe: &'a mut [f32],
    pub ln1w: &'a mut [f32],
    pub ln1b: &'a mut [f32],
    pub qkvw: &'a mut [f32],
    pub qkvb: &'a mut [f32],
    pub attprojw: &'a mut [f32],
    pub attprojb: &'a mut [f32],
    pub ln2w: &'a mut [f32],
    pub ln2b: &'a mut [f32],
    pub fcw: &'a mut [f32],
    pub fcb: &'a mut [f32],
    pub fcprojw: &'a mut [f32],
    pub fcprojb: &'a mut [f32],
    pub lnfw: &'a mut [f32],
    pub lnfb: &'a mut [f32],
}

pub struct ActivationTensors<'a> {
    pub encoded: &'a mut [f32],
    pub ln1: &'a mut [f32],
    pub ln1_mean: &'a mut [f32],
    pub ln1_rstd: &'a mut [f32],
    pub qkv: &'a mut [f32],
    pub atty: &'a mut [f32],
    pub preatt: &'a mut [f32],
    pub att: &'a mut [f32],
    pub attproj: &'a mut [f32],
    pub residual2: &'a mut [f32],
    pub ln2: &'a mut [f32],
    pub ln2_mean: &'a mut [f32],
    pub ln2_rstd: &'a mut [f32],
    pub fch: &'a mut [f32],
    pub fch_gelu: &'a mut [f32],
    pub fcproj: &'a mut [f32],
    pub residual3: &'a mut [f32],
    pub lnf: &'a mut [f32],
    pub lnf_mean: &'a mut [f32],
    pub lnf_rstd: &'a mut [f32],
    pub logits: &'a mut [f32],
    pub probs: &'a mut [f32],
    pub losses: &'a mut [f32],
}

pub struct Gpt2<'a> {
    pub config: Gpt2Config,
    pub params_memory: Vec<f32>,
    pub grads_memory: Vec<f32>,
    pub acts_memory: Vec<f32>,
    pub grads_acts_memory: Vec<f32>,
    pub params: ParameterTensors<'a>,
    pub grads: ParameterTensors<'a>,
    pub acts: ActivationTensors<'a>,
    pub grads_acts: ActivationTensors<'a>,
    pub m_memory: Vec<f32>,
    pub v_memory: Vec<f32>,
    pub batch_size: usize,
    pub seq_len: usize,
    pub inputs: Vec<i32>,
    pub targets: Vec<i32>,
    pub mean_loss: f32,
}

impl<'a> Gpt2<'a> {
    pub fn build_from_checkpoint<P: AsRef<std::path::Path>>(checkpoint_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(checkpoint_path)?;
        let mut model_header = [0i32; 256];
        file.read_exact(bytemuck::bytes_of_mut(&mut model_header))?;

        if model_header[0] != 20240326 {
            return Err("Bad magic model file".into());
        }
        if model_header[1] != 1 {
            return Err("Bad version in model file".into());
        }

        let config = Gpt2Config {
            max_seq_len: model_header[2] as usize,
            vocab_size: model_header[3] as usize,
            num_layers: model_header[4] as usize,
            num_heads: model_header[5] as usize,
            channels: model_header[6] as usize,
        };

        println!("[GPT-2]");
        println!("max_seq_len: {}", config.max_seq_len);
        println!("vocab_size: {}", config.vocab_size);
        println!("num_layers: {}", config.num_layers);
        println!("num_heads: {}", config.num_heads);
        println!("channels: {}", config.channels);

        let param_sizes = [
            config.vocab_size * config.channels, // wte
            config.max_seq_len * config.channels, // wpe
            config.num_layers * config.channels, // ln1w
            config.num_layers * config.channels, // ln1b
            config.num_layers * (3 * config.channels) * config.channels, // qkvw
            config.num_layers * (3 * config.channels), // qkvb
            config.num_layers * config.channels * config.channels, // attprojw
            config.num_layers * config.channels, // attprojb
            config.num_layers * config.channels, // ln2w
            config.num_layers * config.channels, // ln2b
            config.num_layers * (4 * config.channels) * config.channels, // fcw
            config.num_layers * (4 * config.channels), // fcb
            config.num_layers * config.channels * (4 * config.channels), // fcprojw
            config.num_layers * config.channels, // fcprojb
            config.channels, // lnfw
            config.channels, // lnfb
        ];

        let total_params: usize = param_sizes.iter().sum();
        println!("num_parameters: {}", total_params);

        let mut params_memory = vec![0.0f32; total_params];
        file.read_exact(bytemuck::cast_slice_mut(&mut params_memory))?;

        let mut offset = 0;
        let params = ParameterTensors {
            wte: &mut params_memory[offset..offset + param_sizes[0]],
            wpe: &mut params_memory[offset + param_sizes[0]..offset + param_sizes[0] + param_sizes[1]],
            ln1w: &mut params_memory[offset + param_sizes[0] + param_sizes[1]..offset + param_sizes[0] + param_sizes[1] + param_sizes[2]],
            ln1b: &mut params_memory[offset + param_sizes[0] + param_sizes[1] + param_sizes[2]..offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3]],
            qkvw: &mut params_memory[offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3]..offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4]],
            qkvb: &mut params_memory[offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4]..offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5]],
            attprojw: &mut params_memory[offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5]..offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6]],
            attprojb: &mut params_memory[offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6]..offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7]],
            ln2w: &mut params_memory[offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7]..offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8]],
            ln2b: &mut params_memory[offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8]..offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9]],
            fcw: &mut params_memory[offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9]..offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10]],
            fcb: &mut params_memory[offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10]..offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11]],
            fcprojw: &mut params_memory[offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11]..offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11] + param_sizes[12]],
            fcprojb: &mut params_memory[offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11] + param_sizes[12]..offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11] + param_sizes[12] + param_sizes[13]],
            lnfw: &mut params_memory[offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11] + param_sizes[12] + param_sizes[13]..offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11] + param_sizes[12] + param_sizes[13] + param_sizes[14]],
            lnfb: &mut params_memory[offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11] + param_sizes[12] + param_sizes[13] + param_sizes[14]..offset + param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3] + param_sizes[4] + param_sizes[5] + param_sizes[6] + param_sizes[7] + param_sizes[8] + param_sizes[9] + param_sizes[10] + param_sizes[11] + param_sizes[12] + param_sizes[13] + param_sizes[14] + param_sizes[15]],
        };

        Ok(Gpt2 {
            config,
            params_memory,
            grads_memory: Vec::new(),
            acts_memory: Vec::new(),
            grads_acts_memory: Vec::new(),
            params,
            grads: ParameterTensors {
                wte: &mut [], wpe: &mut [], ln1w: &mut [], ln1b: &mut [],
                qkvw: &mut [], qkvb: &mut [], attprojw: &mut [], attprojb: &mut [],
                ln2w: &mut [], ln2b: &mut [], fcw: &mut [], fcb: &mut [],
                fcprojw: &mut [], fcprojb: &mut [], lnfw: &mut [], lnfb: &mut [],
            },
            acts: ActivationTensors {
                encoded: &mut [], ln1: &mut [], ln1_mean: &mut [], ln1_rstd: &mut [],
                qkv: &mut [], atty: &mut [], preatt: &mut [], att: &mut [],
                attproj: &mut [], residual2: &mut [], ln2: &mut [], ln2_mean: &mut [],
                ln2_rstd: &mut [], fch: &mut [], fch_gelu: &mut [], fcproj: &mut [],
                residual3: &mut [], lnf: &mut [], lnf_mean: &mut [], lnf_rstd: &mut [],
                logits: &mut [], probs: &mut [], losses: &mut [],
            },
            grads_acts: ActivationTensors {
                encoded: &mut [], ln1: &mut [], ln1_mean: &mut [], ln1_rstd: &mut [],
                qkv: &mut [], atty: &mut [], preatt: &mut [], att: &mut [],
                attproj: &mut [], residual2: &mut [], ln2: &mut [], ln2_mean: &mut [],
                ln2_rstd: &mut [], fch: &mut [], fch_gelu: &mut [], fcproj: &mut [],
                residual3: &mut [], lnf: &mut [], lnf_mean: &mut [], lnf_rstd: &mut [],
                logits: &mut [], probs: &mut [], losses: &mut [],
            },
            m_memory: Vec::new(),
            v_memory: Vec::new(),
            batch_size: 0,
            seq_len: 0,
            inputs: Vec::new(),
            targets: Vec::new(),
            mean_loss: -1.0,
        })
    }

    pub fn allocate_activations(&mut self, B: usize, T: usize) -> Result<(), String> {
        if !self.acts_memory.is_empty() {
            if B > self.batch_size || T > self.seq_len {
                return Err(format!("Inadequate B or T. Model: {}x{}, Desired: {}x{}", self.batch_size, self.seq_len, B, T));
            }
            return Ok(());
        }

        self.batch_size = B;
        self.seq_len = T;

        let C = self.config.channels;
        let L = self.config.num_layers;
        let NH = self.config.num_heads;
        let V = self.config.vocab_size;

        let act_sizes = [
            B * T * C, // encoded
            L * B * T * C, // ln1
            L * B * T, // ln1_mean
            L * B * T, // ln1_rstd
            L * B * T * 3*C, // qkv
            L * B * T * C, // atty
            L * B * NH * T * T, // preatt
            L * B * NH * T * T, // att
            L * B * T * C, // attproj
            L * B * T * C, // residual2
            L * B * T * C, // ln2
            L * B * T, // ln2_mean
            L * B * T, // ln2_rstd
            L * B * T * 4*C, // fch
            L * B * T * 4*C, // fch_gelu
            L * B * T * C, // fcproj
            L * B * T * C, // residual3
            B * T * C, // lnf
            B * T, // lnf_mean
            B * T, // lnf_rstd
            B * T * V, // logits
            B * T * V, // probs
            B * T, // losses
        ];

        let total_acts: usize = act_sizes.iter().sum();
        println!("num_activations: {}", total_acts);

        self.acts_memory = vec![0.0f32; total_acts];
        self.grads_acts_memory = vec![0.0f32; total_acts]; // Allocate gradients too

        let mut offset = 0;
        self.acts = ActivationTensors {
            encoded: &mut self.acts_memory[offset..offset + act_sizes[0]],
            ln1: &mut self.acts_memory[offset + act_sizes[0]..offset + act_sizes[0] + act_sizes[1]],
            ln1_mean: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2]],
            ln1_rstd: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3]],
            qkv: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4]],
            atty: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5]],
            preatt: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6]],
            att: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7]],
            attproj: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8]],
            residual2: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9]],
            ln2: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10]],
            ln2_mean: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11]],
            ln2_rstd: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12]],
            fch: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13]],
            fch_gelu: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14]],
            fcproj: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15]],
            residual3: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16]],
            lnf: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17]],
            lnf_mean: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18]],
            lnf_rstd: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18] + act_sizes[19]],
            logits: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18] + act_sizes[19]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18] + act_sizes[19] + act_sizes[20]],
            probs: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18] + act_sizes[19] + act_sizes[20]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18] + act_sizes[19] + act_sizes[20] + act_sizes[21]],
            losses: &mut self.acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18] + act_sizes[19] + act_sizes[20] + act_sizes[21]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18] + act_sizes[19] + act_sizes[20] + act_sizes[21] + act_sizes[22]],
        };

        offset = 0;
        self.grads_acts = ActivationTensors {
            encoded: &mut self.grads_acts_memory[offset..offset + act_sizes[0]],
            ln1: &mut self.grads_acts_memory[offset + act_sizes[0]..offset + act_sizes[0] + act_sizes[1]],
            ln1_mean: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2]],
            ln1_rstd: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3]],
            qkv: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4]],
            atty: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5]],
            preatt: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6]],
            att: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7]],
            attproj: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8]],
            residual2: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9]],
            ln2: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10]],
            ln2_mean: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11]],
            ln2_rstd: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12]],
            fch: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13]],
            fch_gelu: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14]],
            fcproj: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15]],
            residual3: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16]],
            lnf: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17]],
            lnf_mean: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18]],
            lnf_rstd: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18] + act_sizes[19]],
            logits: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18] + act_sizes[19]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18] + act_sizes[19] + act_sizes[20]],
            probs: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18] + act_sizes[19] + act_sizes[20]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18] + act_sizes[19] + act_sizes[20] + act_sizes[21]],
            losses: &mut self.grads_acts_memory[offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18] + act_sizes[19] + act_sizes[20] + act_sizes[21]..offset + act_sizes[0] + act_sizes[1] + act_sizes[2] + act_sizes[3] + act_sizes[4] + act_sizes[5] + act_sizes[6] + act_sizes[7] + act_sizes[8] + act_sizes[9] + act_sizes[10] + act_sizes[11] + act_sizes[12] + act_sizes[13] + act_sizes[14] + act_sizes[15] + act_sizes[16] + act_sizes[17] + act_sizes[18] + act_sizes[19] + act_sizes[20] + act_sizes[21] + act_sizes[22]],
        };

        self.inputs = vec![0; B * T];
        self.targets = vec![0; B * T];

        Ok(())
    }

    pub fn forward(&mut self, inputs: &[i32], targets: Option<&[i32]>, B: usize, T: usize) -> Result<(), String> {
        self.allocate_activations(B, T)?;

        self.inputs[..B*T].copy_from_slice(&inputs[..B*T]);
        if let Some(t) = targets {
            self.targets[..B*T].copy_from_slice(&t[..B*T]);
        }

        let C = self.config.channels;
        let L = self.config.num_layers;
        let NH = self.config.num_heads;
        let V = self.config.vocab_size;

        let acts = &mut self.acts;

        crate::layers::encoder_forward(acts.encoded, inputs, self.params.wte, self.params.wpe, B, T, C);

        for l in 0..L {
            let residual = if l == 0 {
                acts.encoded
            } else {
                &acts.residual3[(l-1) * B * T * C..l * B * T * C]
            };

            let l_ln1w = &self.params.ln1w[l * C..(l+1) * C];
            let l_ln1b = &self.params.ln1b[l * C..(l+1) * C];
            let l_qkvw = &self.params.qkvw[l * 3*C * C..(l+1) * 3*C * C];
            let l_qkvb = &self.params.qkvb[l * 3*C..(l+1) * 3*C];
            let l_attprojw = &self.params.attprojw[l * C * C..(l+1) * C * C];
            let l_attprojb = &self.params.attprojb[l * C..(l+1) * C];
            let l_ln2w = &self.params.ln2w[l * C..(l+1) * C];
            let l_ln2b = &self.params.ln2b[l * C..(l+1) * C];
            let l_fcw = &self.params.fcw[l * 4*C * C..(l+1) * 4*C * C];
            let l_fcb = &self.params.fcb[l * 4*C..(l+1) * 4*C];
            let l_fcprojw = &self.params.fcprojw[l * C * 4*C..(l+1) * C * 4*C];
            let l_fcprojb = &self.params.fcprojb[l * C..(l+1) * C];

            let l_ln1 = &mut acts.ln1[l * B * T * C..(l+1) * B * T * C];
            let l_ln1_mean = &mut acts.ln1_mean[l * B * T..(l+1) * B * T];
            let l_ln1_rstd = &mut acts.ln1_rstd[l * B * T..(l+1) * B * T];
            let l_qkv = &mut acts.qkv[l * B * T * 3*C..(l+1) * B * T * 3*C];
            let l_atty = &mut acts.atty[l * B * T * C..(l+1) * B * T * C];
            let l_preatt = &mut acts.preatt[l * B * NH * T * T..(l+1) * B * NH * T * T];
            let l_att = &mut acts.att[l * B * NH * T * T..(l+1) * B * NH * T * T];
            let l_attproj = &mut acts.attproj[l * B * T * C..(l+1) * B * T * C];
            let l_residual2 = &mut acts.residual2[l * B * T * C..(l+1) * B * T * C];
            let l_ln2 = &mut acts.ln2[l * B * T * C..(l+1) * B * T * C];
            let l_ln2_mean = &mut acts.ln2_mean[l * B * T..(l+1) * B * T];
            let l_ln2_rstd = &mut acts.ln2_rstd[l * B * T..(l+1) * B * T];
            let l_fch = &mut acts.fch[l * B * T * 4*C..(l+1) * B * T * 4*C];
            let l_fch_gelu = &mut acts.fch_gelu[l * B * T * 4*C..(l+1) * B * T * 4*C];
            let l_fcproj = &mut acts.fcproj[l * B * T * C..(l+1) * B * T * C];
            let l_residual3 = &mut acts.residual3[l * B * T * C..(l+1) * B * T * C];

            crate::layers::layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
            crate::layers::matmul_forward(l_qkv, l_ln1, l_qkvw, Some(l_qkvb), B, T, C, 3*C);
            crate::layers::attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
            crate::layers::matmul_forward(l_attproj, l_atty, l_attprojw, Some(l_attprojb), B, T, C, C);
            crate::layers::residual_forward(l_residual2, residual, l_attproj, B*T*C);
            crate::layers::layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
            crate::layers::matmul_forward(l_fch, l_ln2, l_fcw, Some(l_fcb), B, T, C, 4*C);
            crate::layers::gelu_forward(l_fch_gelu, l_fch);
            crate::layers::matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, Some(l_fcprojb), B, T, 4*C, C);
            crate::layers::residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
        }

        let final_residual = &acts.residual3[(L-1) * B * T * C..L * B * T * C];
        crate::layers::layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, final_residual, self.params.lnfw, self.params.lnfb, B, T, C);
        crate::layers::matmul_forward(acts.logits, acts.lnf, self.params.wte, None, B, T, C, V);
        crate::layers::softmax_forward(acts.probs, acts.logits, B, T, V);

        if let Some(targets) = targets {
            crate::layers::crossentropy_forward(acts.losses, acts.probs, targets, B, T, V);
            let mean_loss: f32 = acts.losses[..B*T].iter().sum::<f32>() / (B*T) as f32;
            self.mean_loss = mean_loss;
        } else {
            self.mean_loss = -1.0;
        }

        Ok(())
    }

    pub fn zero_grad(&mut self) {
        if !self.grads_memory.is_empty() {
            self.grads_memory.fill(0.0);
        }
        if !self.grads_acts_memory.is_empty() {
            self.grads_acts_memory.fill(0.0);
        }
    }

    pub fn backward(&mut self) {
        if self.mean_loss == -1.0 {
            panic!("Must call forward with targets before backward");
        }

        if self.grads_memory.is_empty() {
            self.grads_memory = vec![0.0; self.params_memory.len()];
            self.grads_acts_memory = vec![0.0; self.acts_memory.len()]; // We already allocated this in forward
            self.zero_grad();
        }

        let B = self.batch_size;
        let T = self.seq_len;
        let C = self.config.channels;
        let L = self.config.num_layers;
        let NH = self.config.num_heads;
        let V = self.config.vocab_size;

        let acts = &self.acts;
        let grads_acts = &mut self.grads_acts;

        let dloss_mean = 1.0 / (B * T) as f32;
        for i in 0..B*T {
            grads_acts.losses[i] = dloss_mean;
        }

        crate::layers::crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, &self.targets, B, T, V);
        crate::layers::matmul_backward(grads_acts.lnf, self.grads.wte, None, grads_acts.logits, acts.lnf, self.params.wte, B, T, C, V);

        let mut dresidual = &mut grads_acts.residual3[(L-1) * B * T * C..L * B * T * C];
        crate::layers::layernorm_backward(dresidual, self.grads.lnfw, self.grads.lnfb, grads_acts.lnf, &acts.residual3[(L-1) * B * T * C..], self.params.lnfw, &acts.lnf_mean, &acts.lnf_rstd, B, T, C);

        for l in (0..L).rev() {
            let residual = if l == 0 {
                acts.encoded
            } else {
                &acts.residual3[(l-1) * B * T * C..l * B * T * C]
            };
            let dresidual_in = if l == 0 {
                &mut grads_acts.encoded
            } else {
                &mut grads_acts.residual3[(l-1) * B * T * C..l * B * T * C]
            };

            let l_ln1w = &self.params.ln1w[l * C..(l+1) * C];
            let l_qkvw = &self.params.qkvw[l * 3*C * C..(l+1) * 3*C * C];
            let l_attprojw = &self.params.attprojw[l * C * C..(l+1) * C * C];
            let l_ln2w = &self.params.ln2w[l * C..(l+1) * C];
            let l_fcw = &self.params.fcw[l * 4*C * C..(l+1) * 4*C * C];
            let l_fcprojw = &self.params.fcprojw[l * C * 4*C..(l+1) * C * 4*C];

            let dl_ln1w = &mut self.grads.ln1w[l * C..(l+1) * C];
            let dl_ln1b = &mut self.grads.ln1b[l * C..(l+1) * C];
            let dl_qkvw = &mut self.grads.qkvw[l * 3*C * C..(l+1) * 3*C * C];
            let dl_qkvb = &mut self.grads.qkvb[l * 3*C..(l+1) * 3*C];
            let dl_attprojw = &mut self.grads.attprojw[l * C * C..(l+1) * C * C];
            let dl_attprojb = &mut self.grads.attprojb[l * C..(l+1) * C];
            let dl_ln2w = &mut self.grads.ln2w[l * C..(l+1) * C];
            let dl_ln2b = &mut self.grads.ln2b[l * C..(l+1) * C];
            let dl_fcw = &mut self.grads.fcw[l * 4*C * C..(l+1) * 4*C * C];
            let dl_fcb = &mut self.grads.fcb[l * 4*C..(l+1) * 4*C];
            let dl_fcprojw = &mut self.grads.fcprojw[l * C * 4*C..(l+1) * C * 4*C];
            let dl_fcprojb = &mut self.grads.fcprojb[l * C..(l+1) * C];

            let l_ln1 = &acts.ln1[l * B * T * C..(l+1) * B * T * C];
            let l_ln1_mean = &acts.ln1_mean[l * B * T..(l+1) * B * T];
            let l_ln1_rstd = &acts.ln1_rstd[l * B * T..(l+1) * B * T];
            let l_qkv = &acts.qkv[l * B * T * 3*C..(l+1) * B * T * 3*C];
            let l_atty = &acts.atty[l * B * T * C..(l+1) * B * T * C];
            let l_att = &acts.att[l * B * NH * T * T..(l+1) * B * NH * T * T];
            let l_residual2 = &acts.residual2[l * B * T * C..(l+1) * B * T * C];
            let l_ln2 = &acts.ln2[l * B * T * C..(l+1) * B * T * C];
            let l_ln2_mean = &acts.ln2_mean[l * B * T..(l+1) * B * T];
            let l_ln2_rstd = &acts.ln2_rstd[l * B * T..(l+1) * B * T];
            let l_fch = &acts.fch[l * B * T * 4*C..(l+1) * B * T * 4*C];
            let l_fch_gelu = &acts.fch_gelu[l * B * T * 4*C..(l+1) * B * T * 4*C];

            let dl_ln1 = &mut grads_acts.ln1[l * B * T * C..(l+1) * B * T * C];
            let dl_qkv = &mut grads_acts.qkv[l * B * T * 3*C..(l+1) * B * T * 3*C];
            let dl_atty = &mut grads_acts.atty[l * B * T * C..(l+1) * B * T * C];
            let dl_preatt = &mut grads_acts.preatt[l * B * NH * T * T..(l+1) * B * NH * T * T];
            let dl_att = &mut grads_acts.att[l * B * NH * T * T..(l+1) * B * NH * T * T];
            let dl_attproj = &mut grads_acts.attproj[l * B * T * C..(l+1) * B * T * C];
            let dl_residual2 = &mut grads_acts.residual2[l * B * T * C..(l+1) * B * T * C];
            let dl_ln2 = &mut grads_acts.ln2[l * B * T * C..(l+1) * B * T * C];
            let dl_fch = &mut grads_acts.fch[l * B * T * 4*C..(l+1) * B * T * 4*C];
            let dl_fch_gelu = &mut grads_acts.fch_gelu[l * B * T * 4*C..(l+1) * B * T * 4*C];
            let dl_fcproj = &mut grads_acts.fcproj[l * B * T * C..(l+1) * B * T * C];
            let dl_residual3 = &mut grads_acts.residual3[l * B * T * C..(l+1) * B * T * C];

            crate::layers::residual_backward(dresidual_in, dl_fcproj, dresidual, B*T*C);
            crate::layers::matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
            crate::layers::gelu_backward(dl_fch, l_fch, dl_fch_gelu);
            crate::layers::matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
            crate::layers::layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
            crate::layers::residual_backward(dresidual_in, dl_attproj, dl_residual2, B*T*C);
            crate::layers::matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
            crate::layers::attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
            crate::layers::matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
            crate::layers::layernorm_backward(dresidual_in, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
        }

        crate::layers::encoder_backward(self.grads.wte, self.grads.wpe, grads_acts.encoded, &self.inputs, B, T, C);
    }

    pub fn update(&mut self, learning_rate: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, t: usize) {
        if self.m_memory.is_empty() {
            self.m_memory = vec![0.0; self.params_memory.len()];
            self.v_memory = vec![0.0; self.params_memory.len()];
        }

        for i in 0..self.params_memory.len() {
            let grad = self.grads_memory[i];
            let param = self.params_memory[i];

            let m = beta1 * self.m_memory[i] + (1.0 - beta1) * grad;
            let v = beta2 * self.v_memory[i] + (1.0 - beta2) * grad * grad;

            let m_hat = m / (1.0 - beta1.powi(t as i32));
            let v_hat = v / (1.0 - beta2.powi(t as i32));

            self.m_memory[i] = m;
            self.v_memory[i] = v;

            self.params_memory[i] -= learning_rate * (m_hat / (v_hat.sqrt() + eps) + weight_decay * param);
        }
    }
}

impl Drop for Gpt2<'_> {
    fn drop(&mut self) {
        // Vecs are automatically dropped.
    }
}
