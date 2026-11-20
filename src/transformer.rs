use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

use crate::embedding::EmbeddingLayer;
use crate::layernorm::Matrix;
use crate::positional::PositionalEncoding;
use crate::transformer_block::{TransformerBlock, TransformerBlockCache};
use crate::tokenizer::Tokenizer;

const CHECKPOINT_MAGIC: &[u8; 8] = b"NRUSTCP1";
const CHECKPOINT_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct Transformer {
    embedding: EmbeddingLayer,
    positional: PositionalEncoding,
    blocks: Vec<TransformerBlock>,
    w_out: Matrix,
    b_out: Vec<f64>,
    grad_w_out: Matrix,
    grad_b_out: Vec<f64>,
    vocab_size: usize,
    embed_dim: usize,
    max_seq_len: usize,
    num_heads: usize,
    ff_dim: usize,
}

impl Transformer {
    pub fn diagnostic_extrema(&self) -> Vec<(String, f64, f64)> {
        let mut out = Vec::new();
        let (weight, grad) = self.embedding.diagnostic_extrema();
        out.push(("EMBEDDING".into(), weight, grad));
        for (index, block) in self.blocks.iter().enumerate() {
            for (name, weight, grad) in block.diagnostic_extrema() {
                out.push((format!("BLOCK_{index}_{name}"), weight, grad));
            }
        }
        out.push(("OUTPUT_PROJECTION".into(), max_abs(self.w_out.iter().flatten().copied().chain(self.b_out.iter().copied())), max_abs(self.grad_w_out.iter().flatten().copied().chain(self.grad_b_out.iter().copied()))));
        out
    }
    pub fn new(vocab_size: usize, embed_dim: usize, num_heads: usize, ff_dim: usize, num_layers: usize, max_seq_len: usize, seed: u64) -> Result<Self, String> {
        if vocab_size == 0 { return Err("vocab_size must be greater than zero".into()); }
        if embed_dim == 0 { return Err("embed_dim must be greater than zero".into()); }
        if num_heads == 0 { return Err("num_heads must be greater than zero".into()); }
        if ff_dim == 0 { return Err("ff_dim must be greater than zero".into()); }
        if !embed_dim.is_multiple_of(num_heads) { return Err("embed_dim must be divisible by num_heads".into()); }
        if max_seq_len == 0 { return Err("max_seq_len must be greater than zero".into()); }
        let embedding = EmbeddingLayer::new(vocab_size, embed_dim, seed);
        let positional = PositionalEncoding::new(max_seq_len, embed_dim)?;
        let mut blocks = Vec::with_capacity(num_layers);
        for layer in 0..num_layers { blocks.push(TransformerBlock::new(embed_dim, num_heads, ff_dim, seed.wrapping_add(layer as u64))?); }
        let mut rng = SplitMix64::new(seed.wrapping_add(num_layers as u64).wrapping_add(0xA076_1D64_78BD_642F));
        let w_out = random_matrix(embed_dim, vocab_size, &mut rng);
        Ok(Self { embedding, positional, blocks, w_out, b_out: vec![0.0; vocab_size], grad_w_out: zeros(embed_dim, vocab_size), grad_b_out: vec![0.0; vocab_size], vocab_size, embed_dim, max_seq_len, num_heads, ff_dim })
    }

    pub fn forward(&self, token_ids: &[usize], numeric_values: &[Option<f64>]) -> Result<Matrix, String> {
        self.forward_with_cache(token_ids, numeric_values).map(|(logits, _, _, _)| logits)
    }

    /// Greedily generates up to `max_new_tokens` from a tokenized prompt.
    /// Returns only newly generated token IDs, including EOS if it is generated.
    pub fn generate(&self, prompt: &[usize], max_new_tokens: usize) -> Result<Vec<usize>, String> {
        if prompt.is_empty() {
            return Err("prompt must contain at least one token".into());
        }

        let mut sequence = prompt.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let numeric_values = vec![None; sequence.len()];
            let logits = self.forward(&sequence, &numeric_values)?;
            let next_token = logits
                .last()
                .and_then(|row| {
                    row.iter()
                        .enumerate()
                        .max_by(|(left_id, left), (right_id, right)| {
                            left.total_cmp(right).then_with(|| right_id.cmp(left_id))
                        })
                        .map(|(token_id, _)| token_id)
                })
                .ok_or_else(|| "model produced no logits for the prompt".to_string())?;
            generated.push(next_token);
            sequence.push(next_token);
            if next_token == Tokenizer::EOS_ID {
                break;
            }
        }
        Ok(generated)
    }

    pub fn forward_with_cache(&self, token_ids: &[usize], numeric_values: &[Option<f64>]) -> Result<(Matrix, Matrix, Vec<TransformerBlockCache>, Matrix), String> {
        if token_ids.len() != numeric_values.len() { return Err("token_ids and numeric_values must have equal lengths".into()); }
        if token_ids.len() > self.max_seq_len { return Err(format!("sequence length {} exceeds max_seq_len {}", token_ids.len(), self.max_seq_len)); }
        let embedding_output = self.embedding.forward_with_numerics(token_ids, numeric_values)?;
        let mut hidden = self.positional.apply(&embedding_output)?;
        let mut block_caches = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (next, cache) = block.forward_with_cache(&hidden)?;
            hidden = next;
            block_caches.push(cache);
        }
        let logits = self.project(&hidden)?;
        Ok((logits, embedding_output, block_caches, hidden))
    }

    fn project(&self, hidden: &[Vec<f64>]) -> Result<Matrix, String> {
        validate(hidden, self.embed_dim, "hidden states")?;
        // W_out is stored row-major as [embedding_dim][vocabulary]. Accumulate
        // one contiguous row at a time; the former per-vocabulary reduction
        // walked down columns and caused cache misses for every output token.
        let mut logits = Vec::with_capacity(hidden.len());
        for row in hidden {
            let mut output = self.b_out.clone();
            for (i, &value) in row.iter().enumerate() {
                for (logit, &weight) in output.iter_mut().zip(&self.w_out[i]) {
                    *logit += value * weight;
                }
            }
            logits.push(output);
        }
        Ok(logits)
    }

    /// Mean next-token negative log-likelihood; the final token has no target.
    pub fn loss(&self, logits: &[Vec<f64>], token_ids: &[usize]) -> f64 {
        if logits.len() != token_ids.len() || logits.iter().any(|r| r.len() != self.vocab_size) || token_ids.iter().any(|&id| id >= self.vocab_size) { return f64::INFINITY; }
        if token_ids.len() < 2 { return 0.0; }
        let mut total = 0.0;
        for (i, row) in logits.iter().take(token_ids.len() - 1).enumerate() {
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_sum_exp = max + row.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
            total += log_sum_exp - row[token_ids[i + 1]];
        }
        total / (token_ids.len() - 1) as f64
    }

    /// Backpropagates mean next-token cross-entropy and accumulates parameter gradients.
    pub fn backward(&mut self, token_ids: &[usize], numeric_values: &[Option<f64>]) -> Result<f64, String> {
        if token_ids.len() != numeric_values.len() { return Err("token_ids and numeric_values must have equal lengths".into()); }
        if token_ids.len() < 2 { return Err("at least two tokens are required for next-token training".into()); }
        let (logits, embedding_output, block_caches, final_hidden) = self.forward_with_cache(token_ids, numeric_values)?;
        let loss = self.loss(&logits, token_ids);
        if !loss.is_finite() { return Err("loss is not finite; check logits and target token IDs".into()); }
        let predictions = token_ids.len() - 1;
        let mut d_logits = zeros(token_ids.len(), self.vocab_size);
        for i in 0..predictions {
            let row = &logits[i];
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum = row.iter().map(|v| (v - max).exp()).sum::<f64>();
            for j in 0..self.vocab_size { d_logits[i][j] = ((row[j] - max).exp() / exp_sum - f64::from(j == token_ids[i + 1])) / predictions as f64; }
        }
        for (hidden, grad) in final_hidden.iter().zip(&d_logits) {
            for (i, &hidden_value) in hidden.iter().enumerate() {
                for (j, &gradient) in grad.iter().enumerate() { self.grad_w_out[i][j] += hidden_value * gradient; }
            }
            for (acc, g) in self.grad_b_out.iter_mut().zip(grad) { *acc += g; }
        }
        let mut d_hidden = zeros(token_ids.len(), self.embed_dim);
        for (pos, logits_grad) in d_logits.iter().enumerate() {
            for (i, grad) in d_hidden[pos].iter_mut().enumerate() {
                *grad = logits_grad.iter().enumerate().map(|(j, value)| value * self.w_out[i][j]).sum();
            }
        }
        let mut block_inputs = Vec::with_capacity(block_caches.len());
        if !block_caches.is_empty() {
            block_inputs.push(self.positional.apply(&embedding_output)?);
            for previous in block_caches.iter().take(block_caches.len() - 1) {
                block_inputs.push(add(&previous.x1, &previous.ff_out)?);
            }
        }
        for index in (0..self.blocks.len()).rev() {
            d_hidden = self.blocks[index].backward(&block_inputs[index], &block_caches[index], &d_hidden)?;
        }
        // Positional encodings are fixed additions, so their derivative is the identity.
        for (position, (&token_id, _)) in token_ids.iter().zip(numeric_values).enumerate() {
            self.embedding.accumulate_grad(token_id, &d_hidden[position])?;
        }
        Ok(loss)
    }

    pub fn apply_gradients(&mut self, learning_rate: f64) -> Result<(), String> {
        if !learning_rate.is_finite() || learning_rate < 0.0 { return Err("learning_rate must be finite and non-negative".into()); }
        self.embedding.apply_gradients(learning_rate)?;
        for block in &mut self.blocks { block.apply_gradients(learning_rate)?; }
        for i in 0..self.embed_dim { for j in 0..self.vocab_size { self.w_out[i][j] -= learning_rate * self.grad_w_out[i][j]; self.grad_w_out[i][j] = 0.0; } }
        for j in 0..self.vocab_size { self.b_out[j] -= learning_rate * self.grad_b_out[j]; self.grad_b_out[j] = 0.0; }
        Ok(())
    }

    pub fn vocab_size(&self) -> usize { self.vocab_size }
    pub fn embedding_dim(&self) -> usize { self.embed_dim }
    pub fn max_seq_len(&self) -> usize { self.max_seq_len }
    pub fn num_heads(&self) -> usize { self.num_heads }
    pub fn ff_dim(&self) -> usize { self.ff_dim }
    pub fn num_layers(&self) -> usize { self.blocks.len() }

    /// Saves model weights, architecture settings, and the exact tokenizer vocabulary.
    /// Save between optimizer updates; accumulated gradients are intentionally not persisted.
    pub fn save_checkpoint(&self, tokenizer: &Tokenizer, path: impl AsRef<Path>) -> Result<(), String> {
        if tokenizer.vocab_size() != self.vocab_size {
            return Err(format!("tokenizer vocabulary size ({}) does not match model vocabulary size ({})", tokenizer.vocab_size(), self.vocab_size));
        }
        let path = path.as_ref();
        let file = File::create(path).map_err(|e| format!("could not create checkpoint '{}': {e}", path.display()))?;
        let mut writer = BufWriter::new(file);
        writer.write_all(CHECKPOINT_MAGIC).map_err(|e| e.to_string())?;
        crate::checkpoint::write_u32(&mut writer, CHECKPOINT_VERSION).map_err(|e| e.to_string())?;
        for value in [self.vocab_size, self.embed_dim, self.num_heads, self.ff_dim, self.blocks.len(), self.max_seq_len] {
            crate::checkpoint::write_usize(&mut writer, value).map_err(|e| e.to_string())?;
        }
        let (configured_max_vocab, corpus_unique_words) = tokenizer.checkpoint_metadata();
        crate::checkpoint::write_usize(&mut writer, configured_max_vocab).map_err(|e| e.to_string())?;
        crate::checkpoint::write_usize(&mut writer, corpus_unique_words).map_err(|e| e.to_string())?;
        for token in tokenizer.checkpoint_vocabulary() {
            crate::checkpoint::write_string(&mut writer, token).map_err(|e| e.to_string())?;
        }
        self.embedding.write_checkpoint_weights(&mut writer).map_err(|e| e.to_string())?;
        for block in &self.blocks {
            block.write_checkpoint_weights(&mut writer).map_err(|e| e.to_string())?;
        }
        crate::checkpoint::write_matrix(&mut writer, &self.w_out).map_err(|e| e.to_string())?;
        crate::checkpoint::write_f64s(&mut writer, &self.b_out).map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| format!("could not finish checkpoint '{}': {e}", path.display()))
    }

    /// Loads a versioned checkpoint and restores its matching tokenizer.
    pub fn load_checkpoint(path: impl AsRef<Path>) -> Result<(Self, Tokenizer), String> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| format!("could not open checkpoint '{}': {e}", path.display()))?;
        let file_len = file.metadata().map_err(|e| e.to_string())?.len();
        let mut reader = BufReader::new(file);
        let mut magic = [0; 8];
        reader.read_exact(&mut magic).map_err(|e| format!("invalid checkpoint header: {e}"))?;
        if &magic != CHECKPOINT_MAGIC { return Err("file is not a Neuro_rust checkpoint".into()); }
        let version = crate::checkpoint::read_u32(&mut reader).map_err(|e| e.to_string())?;
        if version != CHECKPOINT_VERSION { return Err(format!("unsupported checkpoint version {version}")); }
        let vocab_size = crate::checkpoint::read_usize(&mut reader).map_err(|e| e.to_string())?;
        let embed_dim = crate::checkpoint::read_usize(&mut reader).map_err(|e| e.to_string())?;
        let num_heads = crate::checkpoint::read_usize(&mut reader).map_err(|e| e.to_string())?;
        let ff_dim = crate::checkpoint::read_usize(&mut reader).map_err(|e| e.to_string())?;
        let num_layers = crate::checkpoint::read_usize(&mut reader).map_err(|e| e.to_string())?;
        let max_seq_len = crate::checkpoint::read_usize(&mut reader).map_err(|e| e.to_string())?;
        if vocab_size == 0 || embed_dim == 0 || num_heads == 0 || ff_dim == 0 || max_seq_len == 0 || !embed_dim.is_multiple_of(num_heads) {
            return Err("checkpoint contains invalid model dimensions".into());
        }
        let configured_max_vocab = crate::checkpoint::read_usize(&mut reader).map_err(|e| e.to_string())?;
        let corpus_unique_words = crate::checkpoint::read_usize(&mut reader).map_err(|e| e.to_string())?;
        let file_len_usize = usize::try_from(file_len).map_err(|_| "checkpoint file is too large for this platform")?;
        if vocab_size > file_len_usize / 8 { return Err("checkpoint vocabulary size exceeds the file size".into()); }
        let mut vocabulary = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            vocabulary.push(crate::checkpoint::read_string(&mut reader, file_len_usize).map_err(|e| e.to_string())?);
        }
        let tokenizer = Tokenizer::from_checkpoint(vocabulary, configured_max_vocab, corpus_unique_words)?;
        let position = reader.stream_position().map_err(|e| e.to_string())?;
        let expected_weights = checkpoint_weight_count(vocab_size, embed_dim, num_heads, ff_dim, num_layers)?;
        let expected_bytes = u64::try_from(expected_weights).ok().and_then(|n| n.checked_mul(8)).ok_or("checkpoint weight size overflow")?;
        if file_len.checked_sub(position) != Some(expected_bytes) {
            return Err("checkpoint weight data is truncated or has unexpected trailing bytes".into());
        }
        let mut model = Self::new(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len, 0)?;
        model.embedding.read_checkpoint_weights(&mut reader)?;
        for block in &mut model.blocks {
            block.read_checkpoint_weights(&mut reader)?;
        }
        model.w_out = crate::checkpoint::read_matrix(&mut reader, embed_dim, vocab_size).map_err(|e| e.to_string())?;
        model.b_out = crate::checkpoint::read_f64s(&mut reader, vocab_size).map_err(|e| e.to_string())?;
        model.grad_w_out = zeros(embed_dim, vocab_size);
        model.grad_b_out = vec![0.0; vocab_size];
        Ok((model, tokenizer))
    }
}

fn max_abs(values: impl Iterator<Item = f64>) -> f64 { values.map(f64::abs).fold(0.0, f64::max) }

fn checkpoint_weight_count(vocab: usize, dim: usize, heads: usize, ff: usize, layers: usize) -> Result<usize, String> {
    let checked_mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(|| "checkpoint model dimensions overflow".to_string());
    let checked_add = |a: usize, b: usize| a.checked_add(b).ok_or_else(|| "checkpoint model dimensions overflow".to_string());
    let _head_dim = dim.checked_div(heads).ok_or("invalid checkpoint head count")?;
    let embedding_and_output = checked_mul(2, checked_mul(vocab, dim)?)?;
    let per_block = checked_add(
        checked_add(checked_mul(4, checked_mul(dim, dim)?)?, checked_mul(2, checked_mul(dim, ff)?)?)?,
        checked_add(ff, checked_mul(5, dim)?)?,
    )?;
    checked_add(checked_add(embedding_and_output, vocab)?, checked_add(1, checked_mul(layers, per_block)?)?)
}

fn add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Matrix, String> {
    if a.len() != b.len() || a.iter().zip(b).any(|(x, y)| x.len() != y.len()) { return Err("residual matrices must have matching shapes".into()); }
    Ok(a.iter().zip(b).map(|(x,y)| x.iter().zip(y).map(|(u,v)| u+v).collect()).collect())
}
fn validate(x: &[Vec<f64>], width: usize, name: &str) -> Result<(), String> {
    if x.iter().any(|r| r.len() != width) { return Err(format!("{name} rows must have width {width}")); }
    Ok(())
}
fn zeros(m: usize, n: usize) -> Matrix { vec![vec![0.0; n]; m] }
fn random_matrix(m: usize, n: usize, rng: &mut SplitMix64) -> Matrix { (0..m).map(|_| (0..n).map(|_| (rng.next_f64() * 2.0 - 1.0) * 0.05).collect()).collect() }
struct SplitMix64(u64);
impl SplitMix64 { fn new(seed: u64) -> Self { Self(seed) } fn next_f64(&mut self) -> f64 { self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15); let mut z = self.0; z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB); z ^= z >> 31; (z >> 11) as f64 / ((1_u64 << 53) as f64) } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Tokenizer;

    const DEMO_CORPUS: [&str; 9] = ["The mean is 20.43", "The median is 18.00", "An increasing trend remains significant", "One outlier changes the slope", "The trend persists after excluding the outlier", "However the evidence is approximately stable", "The analysis confirms a rising pattern", "Mean and median summarize the values", "The trend is increasing at 20.43"];
    const DEMO: &str = "The increasing trend is significant at 20.43, however excluding -3.7 changes the mean";

    #[test] fn forward_logits_have_sequence_by_vocab_shape() {
        let tokenizer = Tokenizer::build(&DEMO_CORPUS, 2048); let (ids, nums) = tokenizer.tokenize(DEMO);
        let model = Transformer::new(tokenizer.vocab_size(), 32, 2, 64, 1, 64, 17).unwrap();
        let logits = model.forward(&ids, &nums).unwrap();
        assert_eq!(logits.len(), ids.len()); assert!(logits.iter().all(|r| r.len() == tokenizer.vocab_size()));
    }

    #[test] fn random_initial_loss_is_near_log_vocab_size() {
        let tokenizer = Tokenizer::build(&DEMO_CORPUS, 2048); let (ids, nums) = tokenizer.tokenize(DEMO);
        let model = Transformer::new(tokenizer.vocab_size(), 32, 2, 64, 1, 64, 17).unwrap();
        let logits = model.forward(&ids, &nums).unwrap(); let loss = model.loss(&logits, &ids);
        let expected = (tokenizer.vocab_size() as f64).ln();
        assert!(loss > 0.0 && (loss - expected).abs() < 0.2, "loss {loss}, log(vocab) {expected}");
    }

    #[test] fn one_update_reduces_loss_on_training_sequence() {
        let tokenizer = Tokenizer::build(&DEMO_CORPUS, 2048); let (ids, nums) = tokenizer.tokenize(DEMO);
        let mut model = Transformer::new(tokenizer.vocab_size(), 32, 2, 64, 1, 64, 17).unwrap();
        let before = model.loss(&model.forward(&ids, &nums).unwrap(), &ids);
        model.backward(&ids, &nums).unwrap(); model.apply_gradients(0.01).unwrap();
        let after = model.loss(&model.forward(&ids, &nums).unwrap(), &ids);
        println!("Single-step loss: {before:.6} -> {after:.6}"); assert!(after < before, "loss did not decrease: {before} -> {after}");
    }

    #[derive(Clone, Copy, Debug)]
    enum Parameter {
        Embedding(usize, usize),
        OutputWeight(usize, usize),
        OutputBias(usize),
        Block(usize, usize, usize, usize),
    }

    impl Parameter {
        fn value_and_gradient(self, model: &Transformer) -> (f64, f64) {
            match self {
                Self::Embedding(token, dim) => model.embedding.test_parameter_value_and_gradient(token, dim),
                Self::OutputWeight(row, col) => (model.w_out[row][col], model.grad_w_out[row][col]),
                Self::OutputBias(index) => (model.b_out[index], model.grad_b_out[index]),
                Self::Block(block, kind, row, col) => model.blocks[block].test_parameter_value_and_gradient(kind, row, col),
            }
        }
        fn perturb(self, model: &mut Transformer, delta: f64) {
            match self {
                Self::Embedding(token, dim) => model.embedding.test_perturb_parameter(token, dim, delta),
                Self::OutputWeight(row, col) => model.w_out[row][col] += delta,
                Self::OutputBias(index) => model.b_out[index] += delta,
                Self::Block(block, kind, row, col) => model.blocks[block].test_perturb_parameter(kind, row, col, delta),
            }
        }
    }

    fn loss_for(model: &Transformer, ids: &[usize], nums: &[Option<f64>]) -> f64 {
        model.loss(&model.forward(ids, nums).unwrap(), ids)
    }

    #[test]
    fn all_block_and_model_parameter_gradients_match_finite_differences() {
        let ids = [0, 1, 2, 3, 1, 4, 2];
        let nums = vec![None; ids.len()];
        let embed_dim = 128;
        let num_heads = 4;
        let head_dim = embed_dim / num_heads;
        let ff_dim = 512;
        let num_blocks = 4;
        let mut model = Transformer::new(5, embed_dim, num_heads, ff_dim, num_blocks, 8, 23).unwrap();
        model.backward(&ids, &nums).unwrap();

        let mut parameters = Vec::new();
        for token in 0..5 { for dim in [0, embed_dim / 2, embed_dim - 1] { parameters.push(Parameter::Embedding(token, dim)); } }
        for row in [0, embed_dim / 2, embed_dim - 1] { for col in 0..5 { parameters.push(Parameter::OutputWeight(row, col)); } }
        for index in 0..5 { parameters.push(Parameter::OutputBias(index)); }

        // Sample every parameter family in every block at the scaled Phase 2 dimensions.
        for block in 0..num_blocks {
            for head in 0..num_heads {
                for projection in 0..3 {
                    let kind = head * 3 + projection;
                    for (row, col) in [(0, 0), (embed_dim - 1, head_dim - 1)] {
                        parameters.push(Parameter::Block(block, kind, row, col));
                    }
                }
            }
            for (kind, row, col) in [
                (6, 0, 0),
                (6, embed_dim - 1, embed_dim - 1),
                (7, 0, 0),
                (7, embed_dim - 1, ff_dim - 1),
                (8, 0, 0),
                (8, ff_dim - 1, embed_dim - 1),
            ] {
                parameters.push(Parameter::Block(block, kind, row, col));
            }
            for kind in 9..=14 {
                let last = if kind == 9 { ff_dim - 1 } else { embed_dim - 1 };
                for row in [0, last] {
                    parameters.push(Parameter::Block(block, kind, row, 0));
                }
            }
        }

        let epsilon = 1e-5;
        let mut failures = Vec::new();
        let mut max_absolute_error: f64 = 0.0;
        let mut max_relative_error: f64 = 0.0;
        for parameter in parameters.iter().copied() {
            let (_, analytic) = parameter.value_and_gradient(&model);
            let mut plus_model = model.clone(); parameter.perturb(&mut plus_model, epsilon);
            let plus = loss_for(&plus_model, &ids, &nums);
            let mut minus_model = model.clone(); parameter.perturb(&mut minus_model, -epsilon);
            let minus = loss_for(&minus_model, &ids, &nums);
            let numerical = (plus - minus) / (2.0 * epsilon);
            let absolute = (numerical - analytic).abs();
            let relative = absolute / (1e-7 + numerical.abs().max(analytic.abs()));
            max_absolute_error = max_absolute_error.max(absolute);
            max_relative_error = max_relative_error.max(relative);
            if absolute > 2e-6 && relative > 2e-3 { failures.push(format!("{parameter:?}: analytic={analytic:.8e}, numeric={numerical:.8e}, abs={absolute:.3e}, rel={relative:.3e}")); }
        }
        println!("Full Transformer parameter gradient check: {} parameters, max abs error {max_absolute_error:.3e}, max relative error {max_relative_error:.3e}", parameters.len());
        assert!(failures.is_empty(), "parameter gradient mismatches:\n{}", failures.join("\n"));
    }
}
