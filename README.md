# GPT2Rust

A compact, pure Rust reimplementation of the GPT-2 training and inference stack. The
code emphasises explicit tensor operations, manual memory management and an
easy-to-follow control flow so that the complete transformer pipeline can be read,
modified and experimented with without relying on external deep-learning crates.

## Table of Contents
- [Project Objectives](#project-objectives)
- [Repository Map](#repository-map)
- [Component Deep Dive](#component-deep-dive)
  - [Low level kernels (`src/layers.rs`)](#low-level-kernels-srclayersrs)
  - [Model orchestration (`src/model.rs`)](#model-orchestration-srcmodelrs)
  - [Streaming data loader (`src/loader.rs`)](#streaming-data-loader-srcloaderrs)
  - [Executable entry point (`src/main.rs`)](#executable-entry-point-srcmainrs)
  - [Dataset bootstrap script (`downloaddataset.sh`)](#dataset-bootstrap-script-downloaddatasetsh)
- [Execution Flow](#execution-flow)
- [Model Architecture Overview](#model-architecture-overview)
- [Data and Checkpoint Files](#data-and-checkpoint-files)
- [Building and Running](#building-and-running)
- [Extending the Project](#extending-the-project)
- [Current Gaps & TODOs](#current-gaps--todos)
- [Further Reading](#further-reading)

## Project Objectives

The repository mirrors the C reference implementation that ships with Andrej
Karpathy's GPT-2 micrograd projects, but reimagined in idiomatic Rust. The focus is
on:

* **Transparency** – every tensor operation is written out and easy to debug.
* **Determinism** – random number generation and optimisation are fully controlled.
* **Experimentation** – the model struct exposes memory buffers that can be modified
  or instrumented for research.

## Repository Map

| Path | Description |
|------|-------------|
| `Cargo.toml` | Crate metadata and dependency declarations. |
| `README.md` | This guide. |
| `downloaddataset.sh` | Convenience script that downloads weights and token files from the LLMC starter pack. |
| `src/layers.rs` | Collection of stateless tensor kernels (attention, layer norm, GELU, matmul, etc.). |
| `src/loader.rs` | Mini-batch streaming loader for memory-mapped token datasets. |
| `src/main.rs` | Binary entry point containing the training loop, evaluation and sampling code. |
| `src/model.rs` | High-level transformer assembly that wires kernels together and manages parameters/activations. |

## Component Deep Dive

### Low level kernels (`src/layers.rs`)

This module contains the raw mathematical primitives. They operate on contiguous
slices (`&[f32]`/`&mut [f32]`) with no heap allocation.

* **GELU activation** – implemented twice: a scalar fallback and an AVX2 accelerated
  version toggled at runtime via `is_x86_feature_detected!("avx2")`. Both use a
  quintic polynomial approximation for speed.
* **Embedding encoder** – adds learned token (`wte`) and positional (`wpe`) vectors
  for each token in a batch.
* **Layer Normalisation** – forward and backward passes with explicit reduction
  loops tracking per-token mean and reciprocal standard deviation.
* **Matrix multiplication** – dense linear layers with optional bias handling,
  implemented in terms of nested loops to keep data dependencies explicit.
* **Self-attention** – causal multi-head attention forward and backward passes. The
  implementation materialises `preatt` (scaled dot products before softmax) and
  `att` (softmaxed weights) buffers to keep intermediate results for the backward
  sweep.
* **Residual connections** – element-wise additions used throughout the transformer
  blocks.
* **Softmax & Cross-entropy** – numerically stable softmax followed by loss and
  gradient helpers tailored to autoregressive language modelling.

The module is intentionally stateless; all mutable state is owned by the caller in
`model.rs`.

### Model orchestration (`src/model.rs`)

`model.rs` provides the `Gpt2` struct which owns every tensor buffer required for
training, evaluation and sampling:

* **Configuration (`Gpt2Config`)** – derived from a checkpoint header and contains
  model hyper-parameters (sequence length, vocabulary size, number of layers,
  heads and embedding channels).
* **ParameterTensors / ActivationTensors** – typed views that slice the underlying
  `Vec<f32>` buffers into semantically meaningful ranges. Separate instances exist
  for parameters, gradients, activations and activation gradients. This keeps the
  borrow checker satisfied while still operating on flat arrays.
* **`build_from_checkpoint`** – reads a binary checkpoint (`gpt2_124M.bin`) using a
  lightweight header (magic number, version and configuration). Parameters are
  memory-mapped into the `params_memory` buffer and sliced into the `params`
  struct.
* **`allocate_activations`** – lazily allocates activation/gradient buffers for a
  requested batch size `B` and sequence length `T`, reusing existing memory if the
  request fits in previously allocated space. This function computes the exact size
  of each intermediate tensor (e.g. per-layer QKV projections, attention scores,
  feed-forward activations) so that the rest of the code can index safely.
* **`forward`** – orchestrates one pass through the network. It runs the embedding
  stage, each transformer block (layer norm → QKV projection → attention →
  projection → residual → MLP → GELU → projection → residual) and finally projects
  onto vocabulary logits before computing probabilities and optional losses.
* **`zero_grad`** – clears parameter and activation gradients.
* **`backward`** – mirrors the forward pass in reverse, distributing gradients to
  every parameter tensor using the helpers in `layers.rs`.
* **`update`** – applies an Adam-style update with decoupled weight decay, keeping
  first (`m_memory`) and second (`v_memory`) moment buffers.

All tensors remain on the CPU. There are no external BLAS dependencies; everything
runs through Rust loops.

### Streaming data loader (`src/loader.rs`)

`DataLoader` streams contiguous 32-bit token IDs from binary files. It tracks the
current file position, rewinds when the end is reached and exposes three buffers
per batch:

* `batch` – raw tokens read from disk (with an extra token for the next-step target).
* `inputs` – the slice fed into the model.
* `targets` – the input shifted by one token, used for the autoregressive loss.

Datasets must be stored as little-endian `i32` values produced by the original
Python scripts in the LLMC starter pack.

### Executable entry point (`src/main.rs`)

The binary stitches everything together:

1. Loads a GPT-2 checkpoint (`gpt2_124M.bin`) located next to the executable.
2. Scans the `data/` directory for matching `*_train.bin`/`*_val.bin` pairs.
3. Instantiates a `DataLoader` for the chosen dataset (currently the first match).
4. Runs a simple training loop:
   * Every tenth step the model is evaluated on `val_num_batches` batches.
   * Every twentieth step tokens are sampled autoregressively to inspect progress.
   * Each iteration executes `forward → zero_grad → backward → update`.
5. Reports loss values and iteration timings.

`random_u32`, `random_f32` and `sample_mult` form a tiny xorshift RNG used during
sampling to convert probability distributions into discrete token IDs.

### Dataset bootstrap script (`downloaddataset.sh`)

Shell script that downloads:

* Pre-trained GPT-2 weights and auxiliary debug snapshots.
* Tokeniser binary.
* Token datasets (`tiny_shakespeare_*`, `hellaswag_*`).

Files prefixed with `tiny_shakespeare` or `hellaswag` are saved under
`data/tinyshakespeare/` and `data/hellaswag/` respectively; other files go to the
repository root. To use them with the current Rust binary either move or symlink the
train/val files into `data/` so they follow the `<name>_train.bin` / `<name>_val.bin`
pattern that `main.rs` expects.

## Execution Flow

```
+------------------+      +------------------+      +----------------------+      +-----------------+
| downloaddataset  |      | DataLoader       |      | Gpt2::forward        |      | Training Loop   |
| (prepare data)   +----->+ (stream tokens)  +----->+ (run transformer)    +----->+ (backward+Adam) |
+------------------+      +------------------+      +----------------------+      +-----------------+
                                                              |
                                                              v
                                                     logits / losses / samples
```

At runtime the program performs the following steps per iteration:

1. `DataLoader::next_batch` fills `inputs`/`targets` with `B*T` token IDs.
2. `Gpt2::forward` computes logits and losses, storing every intermediate tensor.
3. `Gpt2::zero_grad` clears buffers.
4. `Gpt2::backward` accumulates gradients for parameters and activations.
5. `Gpt2::update` applies Adam updates using the configured learning rate and
   momentum hyper-parameters.
6. Optional: `main.rs` prints validation metrics or samples new text by repeatedly
   calling `forward` with a growing prompt.

## Model Architecture Overview

* **Embeddings** – learned token (`wte`) and positional (`wpe`) tables of size
  `vocab_size × channels` and `max_seq_len × channels`.
* **Transformer blocks** (`num_layers` repetitions):
  1. Pre-attention layer norm (`ln1w`, `ln1b`).
  2. Linear projection into query/key/value (`qkvw`, `qkvb`).
  3. Multi-head self-attention with causal masking (`num_heads`).
  4. Linear projection back to the model dimension (`attprojw`, `attprojb`).
  5. Residual addition.
  6. Second layer norm (`ln2w`, `ln2b`).
  7. Feed-forward MLP with hidden size `4 * channels` (`fcw`, `fcb`).
  8. GELU activation (polynomial approximation).
  9. Projection back to the model dimension (`fcprojw`, `fcprojb`).
  10. Residual addition.
* **Final layer norm** (`lnfw`, `lnfb`).
* **LM head** – ties weights with `wte` to produce vocabulary logits.
* **Loss** – cross-entropy over the shifted targets.

## Data and Checkpoint Files

* **Checkpoint (`gpt2_124M.bin`)** – binary file with a 256 `i32` header followed by
  raw `f32` parameters. Mandatory for `Gpt2::build_from_checkpoint`.
* **Token datasets (`*_train.bin`, `*_val.bin`)** – contiguous `i32` token IDs. The
  program auto-discovers dataset pairs under `data/`.
* **Tokenizer (`gpt2_tokenizer.bin`)** – not consumed by the Rust binary yet, but
  useful if you want to decode generated tokens externally.

## Building and Running

```bash
# Fetch dependencies and compile in release mode
cargo build --release

# Run training/evaluation (expects checkpoints and data files to be in place)
cargo run --release
```

The executable prints available datasets, picks the first one, then starts a
40-iteration training loop with batch size `B = 4` and sequence length `T = 64`.

### Preparing data

```bash
# Download weights and datasets (requires curl)
./downloaddataset.sh

# Move or link the desired token files into data/
ln -s data/tinyshakespeare/tiny_shakespeare_train.bin data/tiny_shakespeare_train.bin
ln -s data/tinyshakespeare/tiny_shakespeare_val.bin data/tiny_shakespeare_val.bin
```

Ensure that the final directory contains files named `<dataset>_train.bin` and
`<dataset>_val.bin`, otherwise `main.rs` will not discover them.

### Runtime configuration

Key hyper-parameters currently live in `src/main.rs`:

* `B` – batch size (default 4).
* `T` – sequence length (default 64 tokens).
* `val_num_batches` – number of validation batches per evaluation (default 10).
* Training steps – fixed at 40 iterations for demonstration purposes.

Modify the file or extend the binary with CLI arguments for more flexibility.

## Extending the Project

* **Alternate activations or optimisers** – swap out the GELU polynomial or replace
  Adam by adjusting `layers.rs` and `Gpt2::update`.
* **Different checkpoints** – `build_from_checkpoint` reads dimensions from the
  header, so any compatible binary with the same format should load. Adjust the
  download script if needed.
* **Token decoding** – integrate a tokenizer to turn sampled token IDs into human
  readable text.
* **Dataset selection** – add command-line parsing to choose among multiple dataset
  pairs discovered under `data/`.
* **Instrumentation** – the flat buffers make it easy to insert logging, gradient
  checks or export functionality.

## Current Gaps & TODOs

* The download script nests datasets under `data/<name>/`; either adjust the script
  or move files so that the auto-discovery logic can find them.
* `main.rs` always selects the first dataset; command-line arguments or an
  interactive prompt would make experimentation easier.
* There is no tokenizer integration, so generated token IDs need to be decoded with
  external tooling.
* The AVX2 kernels target `x86_64`; other architectures fall back to the scalar
  versions but may see reduced performance.
* Parameter/activation buffers assume enough host memory for the configured
  sequence length; add checks if you plan to scale up.

## Further Reading

* [GPT-2 paper (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* [Andrej Karpathy – nanoGPT / llmc projects](https://github.com/karpathy/nanoGPT)
* [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
