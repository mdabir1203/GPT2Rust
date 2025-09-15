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

        let batch = vec![0i32; B * T + 1];
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{remove_file, File};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_file(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("{}_{}", name, nanos));
        path
    }

    fn write_i32_file(path: &PathBuf, values: &[i32]) {
        let mut file = File::create(path).unwrap();
        for &v in values {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn dataloader_new_and_next_batch_reads_expected_sequences() {
        let path = temp_file("tokens");
        let values = [1, 2, 3, 4, 5, 6];
        write_i32_file(&path, &values);

        let B = 1;
        let T = 2;
        let mut loader = DataLoader::new(&path, B, T).unwrap();

        assert_eq!(loader.batch.len(), B * T + 1);
        assert_eq!(loader.inputs.len(), B * T);
        assert_eq!(loader.targets.len(), B * T);
        assert!(loader.num_batches >= 1);

        loader.next_batch().unwrap();
        assert_eq!(&loader.inputs, &[1, 2]);
        assert_eq!(&loader.targets, &[2, 3]);

        loader.next_batch().unwrap();
        assert_eq!(&loader.inputs, &[3, 4]);
        assert_eq!(&loader.targets, &[4, 5]);

        drop(loader);
        remove_file(&path).unwrap();
    }

    #[test]
    fn dataloader_reset_resets_position() {
        let path = temp_file("tokens_reset");
        let values = [10, 11, 12, 13, 14, 15];
        write_i32_file(&path, &values);

        let B = 1;
        let T = 2;
        let mut loader = DataLoader::new(&path, B, T).unwrap();

        loader.next_batch().unwrap();
        assert_eq!(&loader.inputs, &[10, 11]);
        assert!(loader.current_position > 0);

        loader.next_batch().unwrap();
        assert_ne!(&loader.inputs, &[10, 11]);

        loader.reset();
        assert_eq!(loader.current_position, 0);

        loader.next_batch().unwrap();
        assert_eq!(&loader.inputs, &[10, 11]);

        drop(loader);
        remove_file(&path).unwrap();
    }
}
