# gpt2-rs

A compact, pure Rust implementation of a GPT-2 style transformer intended for
experimentation and education.  The project mirrors the layout of the original C
code base the user provided while embracing idiomatic Rust constructs such as
modules, structs and unit tests.

## Features

- Deterministic weight initialisation through a tiny xorshift PRNG.
- Naïve but easy to follow implementations of multi-head self-attention,
  layer-normalisation and the feed-forward network.
- Minimal command line application that generates new text given a prompt.
- Unit tests covering the most important tensor operations and model utilities.

## Running the example

```
cargo run --release -- "Rust is"
```

Use `--steps <n>` to control how many tokens should be generated in addition to
the provided prompt.

## Project layout

```
├── Cargo.toml        # crate metadata
├── src
│   ├── layers.rs     # individual building blocks (attention, MLP, ...)
│   ├── loader.rs     # deterministic initialisation helpers
│   ├── model.rs      # high level GPT-2 inspired model
│   └── main.rs       # command line entry point
└── README.md
```

The implementation purposefully keeps the mathematics explicit which makes it
a good starting point for experimenting with alternative architectures or
sampling strategies without having to depend on external libraries.
