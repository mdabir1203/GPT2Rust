mod layers;
mod loader;
mod model;

use libc::{clock_gettime, timespec, CLOCK_MONOTONIC};
use model::{Gpt2Config, Gpt2Model};

fn monotonic_time() -> timespec {
    unsafe {
        let mut ts = timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let result = clock_gettime(CLOCK_MONOTONIC, &mut ts);
        if result != 0 {
            panic!("clock_gettime failed: {}", std::io::Error::last_os_error());
        }
        ts
    }
}

fn elapsed_ms(start: timespec, end: timespec) -> f64 {
    let mut secs = end.tv_sec - start.tv_sec;
    let mut nanos = end.tv_nsec - start.tv_nsec;
    if nanos < 0 {
        secs -= 1;
        nanos += 1_000_000_000;
    }
    secs as f64 * 1_000.0 + nanos as f64 / 1_000_000.0
}

fn parse_args() -> (String, usize) {
    let mut args = std::env::args().skip(1);
    let mut steps = 32usize;
    let mut prompt_parts = Vec::new();
    while let Some(arg) = args.next() {
        if arg == "--steps" {
            if let Some(value) = args.next() {
                if let Ok(parsed) = value.parse::<usize>() {
                    steps = parsed;
                }
            }
        } else {
            prompt_parts.push(arg);
        }
    }
    let prompt = if prompt_parts.is_empty() {
        String::from("Hello, GPT-2!")
    } else {
        prompt_parts.join(" ")
    };
    (prompt, steps)
}

fn main() {
    let (prompt, steps) = parse_args();
    let config = Gpt2Config::small();
    let model = Gpt2Model::initialise(config, 0x_cafebabedeadbeefu64);

    let start = monotonic_time();
    let generated = model.generate(&prompt, steps);
    let end = monotonic_time();

    println!("Prompt   : {}", prompt);
    println!("Generated: {}", generated);
    println!("Steps    : {}", steps);
    println!("Duration : {:.3} ms", elapsed_ms(start, end));
}
