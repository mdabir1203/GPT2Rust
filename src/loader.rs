use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub struct DataLoader {
    pub B: usize,
    pub T: usize,
    tokens_file: File,
    file_size: u64,
    current_position: u64,
    pub batch: Vec<i32>,
    pub inputs: Vec<i32>,
    pub targets: Vec<i32>,
    pub num_batches: usize,
}

impl DataLoader {
    pub fn new<P: AsRef<Path>>(
        filename: P,
        B: usize,
        T: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut tokens_file = File::open(filename)?;
        let file_size = tokens_file.seek(SeekFrom::End(0))?;
        tokens_file.seek(SeekFrom::Start(0))?;

        if file_size < ((B * T + 1) * std::mem::size_of::<i32>()) as u64 {
            return Err("File too small".into());
        }

        let mut batch = vec![0i32; B * T + 1];
        let inputs = batch[0..B * T].to_vec();
        let targets = batch[1..B * T + 1].to_vec();

        Ok(DataLoader {
            B,
            T,
            tokens_file,
            file_size,
            current_position: 0,
            batch,
            inputs,
            targets,
            num_batches: (file_size as usize) / (B * T * std::mem::size_of::<i32>()),
        })
    }

    pub fn reset(&mut self) {
        self.current_position = 0;
    }

    pub fn next_batch(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.current_position + ((self.B * self.T + 1) * std::mem::size_of::<i32>()) as u64
            > self.file_size
        {
            self.current_position = 0;
        }

        self.tokens_file
            .seek(SeekFrom::Start(self.current_position))?;
        let bytes_to_read = (self.B * self.T + 1) * std::mem::size_of::<i32>();
        let mut buffer = vec![0u8; bytes_to_read];
        self.tokens_file.read_exact(&mut buffer)?;

        // Convert bytes to i32 (assuming little-endian)
        for (i, chunk) in buffer.chunks_exact(std::mem::size_of::<i32>()).enumerate() {
            self.batch[i] = i32::from_le_bytes(chunk.try_into().unwrap());
        }

        self.inputs.copy_from_slice(&self.batch[0..self.B * self.T]);
        self.targets
            .copy_from_slice(&self.batch[1..self.B * self.T + 1]);

        self.current_position += (self.B * self.T * std::mem::size_of::<i32>()) as u64;

        Ok(())
    }
}

impl Drop for DataLoader {
    fn drop(&mut self) {
        // File is automatically closed when `tokens_file` is dropped.
    }
}
