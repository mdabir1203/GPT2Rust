mod layers;
mod model;
mod loader;

use model::Gpt2;
use loader::DataLoader;
use std::time::{Instant};
use std::path::Path;

const GPT2_EOT: i32 = 50256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Gpt2::build_from_checkpoint("gpt2_124M.bin")?;

    let train_tokens = if Path::new("data/tiny_shakespeare_train.bin").exists() {
        "data/tiny_shakespeare_train.bin"
    } else {
        "data/TinyStories_train.bin"
    };
    let val_tokens = if Path::new("data/tiny_shakespeare_val.bin").exists() {
        "data/tiny_shakespeare_val.bin"
    } else {
        "data/TinyStories_val.bin"
    };

    let B = 4;
    let T = 64;

    let mut train_loader = DataLoader::new(train_tokens, B, T)?;
    println!("train dataset num_batches: {}", train_loader.num_batches);

    let mut val_loader = DataLoader::new(val_tokens, B, T)?;
    println!("val dataset num_batches: {}", val_loader.num_batches);
    let val_num_batches = 10;

    // RNG for sampling
    let mut rng_state: u64 = 1337;
    const GEN_MAX_LENGTH: usize = 64;
    let mut gen_tokens = vec![0i32; GEN_MAX_LENGTH];

    for step in 0..=40 {
        // Validation
        if step % 10 == 0 {
            let mut val_loss = 0.0f32;
            val_loader.reset();
            for _ in 0..val_num_batches {
                val_loader.next_batch()?;
                model.forward(&val_loader.inputs, Some(&val_loader.targets), B, T)?;
                val_loss += model.mean_loss;
            }
            println!("step {}: val loss {}", step, val_loss / val_num_batches as f32);
        }

        // Sampling
        if step > 0 && step % 20 == 0 {
            gen_tokens[0] = GPT2_EOT;
            for t in 1..GEN_MAX_LENGTH {
                model.forward(&gen_tokens[..t], None, 1, t)?;
                let probs = &model.acts.probs[(t-1) * model.config.vocab_size..t * model.config.vocab_size];
                let coin = random_f32(&mut rng_state);
                let next_token = sample_mult(probs, model.config.vocab_size, coin) as i32;
                gen_tokens[t] = next_token;
            }
            print!("generated: ");
            for &token in &gen_tokens[..] {
                print!("{} ", token);
            }
            println!();
        }

        // Training Step
        let start = Instant::now();
        train_loader.next_batch()?;
        model.forward(&train_loader.inputs, Some(&train_loader.targets), B, T)?;
        model.zero_grad();
        model.backward();
        model.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step + 1);
        let duration = start.elapsed();

        println!("step {}: train loss {} (took {} ms)", step, model.mean_loss, duration.as_millis());
    }

    Ok(())
}

// Simple RNG (Xorshift)
fn random_u32(state: &mut u64) -> u32 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ((*state).wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32
}

fn random_f32(state: &mut u64) -> f32 {
    (random_u32(state) >> 8) as f32 / 16777216.0
}

fn sample_mult(probabilities: &[f32], n: usize, coin: f32) -> usize {
    let mut cdf = 0.0;
    for (i, &p) in probabilities.iter().enumerate() {
        cdf += p;
        if coin < cdf {
            return i;
        }
    }
    n - 1
}

    println!("Prompt   : {}", prompt);
    println!("Generated: {}", generated);
    println!("Steps    : {}", steps);
    println!("Duration : {:.3} ms", elapsed_ms(start, end));
}
