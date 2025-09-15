mod layers;
mod loader;
#[cfg(not(test))]
mod model;

#[cfg(not(test))]
use model::Gpt2;
#[cfg(not(test))]
use loader::DataLoader;
#[cfg(not(test))]
use std::fs;
#[cfg(not(test))]
use std::path::{Path, PathBuf};
#[cfg(not(test))]
use std::time::Instant;

#[cfg(not(test))]
const GPT2_EOT: i32 = 50256;

#[cfg(not(test))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Gpt2::build_from_checkpoint("gpt2_124M.bin")?;

    // --- START: Dynamic Dataset Discovery ---
    let data_dir = Path::new("data");
    if !data_dir.exists() {
        return Err(format!("Data directory '{}' not found", data_dir.display()).into());
    }

    // Read all entries in the data directory
    let entries = fs::read_dir(data_dir)?;

    // Find all files ending with "_train.bin"
    let mut datasets: Vec<(String, PathBuf, PathBuf)> = Vec::new(); // (name, train_path, val_path)

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.ends_with("_train.bin") {
                    // Extract the dataset name (remove "_train.bin")
                    let dataset_name = filename.strip_suffix("_train.bin").unwrap_or(filename).to_string();

                    // Construct expected validation file path
                    let val_filename = format!("{}_val.bin", dataset_name);
                    let val_path = data_dir.join(&val_filename);

                    // Check if validation file exists
                    if val_path.exists() {
                        datasets.push((dataset_name, path, val_path));
                    } else {
                        eprintln!("Warning: Found train file '{}' but no corresponding validation file '{}'. Skipping.", path.display(), val_path.display());
                    }
                }
            }
        }
    }

    if datasets.is_empty() {
        return Err("No valid datasets found. Expected files like 'dataset_name_train.bin' and 'dataset_name_val.bin' in the 'data/' directory.".into());
    }

    // Print available datasets
    println!("Available datasets:");
    for (i, (name, train_path, val_path)) in datasets.iter().enumerate() {
        println!("  {}: {}", i + 1, name);
        println!("    Train: {}", train_path.display());
        println!("    Val:   {}", val_path.display());
    }

    // Choose the first dataset by default
    // TODO: Extend this to accept a command-line argument for dataset selection
    let chosen_idx = 0;
    let (dataset_name, train_path, val_path) = &datasets[chosen_idx];
    println!("\nUsing dataset: '{}'", dataset_name);
    // --- END: Dynamic Dataset Discovery ---

    let B = 4;
    let T = 64;

    let mut train_loader = DataLoader::new(train_path, B, T)?;
    println!("train dataset num_batches: {}", train_loader.num_batches);

    let mut val_loader = DataLoader::new(val_path, B, T)?;
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
            print!("step {} generated: ", step);
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

#[cfg(test)]
mod tests {
    #[test]
    fn random_u32_matches_reference_implementation() {
        let mut state = 123_456_789u64;
        let mut reference_state = state;
        reference_state ^= reference_state >> 12;
        reference_state ^= reference_state << 25;
        reference_state ^= reference_state >> 27;
        let expected = ((reference_state.wrapping_mul(0x2545F4914F6CDD1D)) >> 32) as u32;

        let value = super::random_u32(&mut state);
        assert_eq!(value, expected);
        assert_eq!(state, reference_state);
    }

    #[test]
    fn random_f32_produces_value_in_unit_interval() {
        let mut state = 987_654_321u64;
        let mut reference_state = state;
        reference_state ^= reference_state >> 12;
        reference_state ^= reference_state << 25;
        reference_state ^= reference_state >> 27;
        let rand = ((reference_state.wrapping_mul(0x2545F4914F6CDD1D)) >> 32) as u32;
        let expected = ((rand >> 8) as f32) / 16_777_216.0;

        let value = super::random_f32(&mut state);
        assert!(value >= 0.0 && value < 1.0);
        assert!((value - expected).abs() < 1e-7);
        assert_eq!(state, reference_state);
    }

    #[test]
    fn sample_mult_selects_based_on_cumulative_probability() {
        let probabilities = [0.1, 0.2, 0.7];
        assert_eq!(super::sample_mult(&probabilities, probabilities.len(), 0.05), 0);
        assert_eq!(super::sample_mult(&probabilities, probabilities.len(), 0.1), 1);
        assert_eq!(super::sample_mult(&probabilities, probabilities.len(), 0.3), 2);
        assert_eq!(super::sample_mult(&probabilities, probabilities.len(), 0.95), 2);
    }
}
