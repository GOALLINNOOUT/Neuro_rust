use std::io::{Read, Write};

use crate::attention::AttentionCache;
use crate::feedforward::FeedForward;
use crate::layernorm::{LayerNorm, Matrix};
use crate::multihead::MultiHeadAttention;

#[derive(Debug, Clone)]
pub struct TransformerBlock {
    attention_norm: LayerNorm,
    attention: MultiHeadAttention,
    feedforward_norm: LayerNorm,
    feedforward: FeedForward,
    embed_dim: usize,
}

#[derive(Debug, Clone)]
pub struct TransformerBlockCache {
    pub ln1_out: Matrix,
    pub ln1_xnorm: Matrix,
    pub attn_out: Matrix,
    pub attn_caches: Vec<AttentionCache>,
    pub concat_heads: Matrix,
    pub x1: Matrix,
    pub ln2_out: Matrix,
    pub ln2_xnorm: Matrix,
    pub ff_out: Matrix,
    pub ff_hidden: Matrix,
}

impl TransformerBlock {
    pub(crate) fn diagnostic_extrema(&self) -> Vec<(String, f64, f64)> {
        let mut out = Vec::new();
        for (name, weight, grad) in self.attention.diagnostic_extrema() {
            out.push((format!("ATTN_{name}"), weight, grad));
        }
        for (name, weight, grad) in self.feedforward.diagnostic_extrema() { out.push((name.to_string(), weight, grad)); }
        let (weight, grad) = self.attention_norm.diagnostic_extrema();
        out.push(("ATTN_NORM".into(), weight, grad));
        let (weight, grad) = self.feedforward_norm.diagnostic_extrema();
        out.push(("FFN_NORM".into(), weight, grad));
        out
    }
    pub fn new(embed_dim: usize, num_heads: usize, ff_dim: usize, seed: u64) -> Result<Self, String> {
        Ok(Self {
            attention_norm: LayerNorm::new(embed_dim),
            attention: MultiHeadAttention::new(embed_dim, num_heads, seed)?,
            feedforward_norm: LayerNorm::new(embed_dim),
            feedforward: FeedForward::new(embed_dim, ff_dim, seed.wrapping_add(1)),
            embed_dim,
        })
    }

    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Matrix, String> {
        self.forward_with_cache(x).map(|(out, _)| out)
    }

    pub fn forward_with_cache(&self, x: &[Vec<f64>]) -> Result<(Matrix, TransformerBlockCache), String> {
        validate(x, self.embed_dim, "input")?;
        let (ln1_out, ln1_xnorm) = self.attention_norm.forward(x);
        let (attn_out, attn_caches, concat_heads) = self.attention.forward_with_cache(&ln1_out)?;
        let x1 = add(x, &attn_out)?;
        let (ln2_out, ln2_xnorm) = self.feedforward_norm.forward(&x1);
        let (ff_out, ff_hidden) = self.feedforward.forward_with_cache(&ln2_out);
        let output = add(&x1, &ff_out)?;
        let cache = TransformerBlockCache { ln1_out, ln1_xnorm, attn_out, attn_caches, concat_heads, x1, ln2_out, ln2_xnorm, ff_out, ff_hidden };
        Ok((output, cache))
    }

    pub fn backward(&mut self, x: &[Vec<f64>], cache: &TransformerBlockCache, d_output: &[Vec<f64>]) -> Result<Matrix, String> {
        validate(x, self.embed_dim, "input")?;
        validate(d_output, self.embed_dim, "output gradient")?;
        // The residual contributes this gradient unchanged around the FFN.
        let d_x1_residual = d_output.to_vec();
        let d_ff = self.feedforward.backward(&cache.ln2_out, &cache.ff_hidden, &d_x1_residual);
        let d_ln2 = self.feedforward_norm.backward(&cache.x1, &cache.ln2_xnorm, &d_ff);
        // Addition sends gradients down both branches, so sum residual and sublayer paths.
        let d_x1 = add(&d_x1_residual, &d_ln2)?;
        let d_attn = self.attention.backward(&cache.ln1_out, &cache.attn_caches, &cache.concat_heads, &d_x1)?;
        let d_ln1 = self.attention_norm.backward(x, &cache.ln1_xnorm, &d_attn);
        add(&d_x1, &d_ln1)
    }

    pub fn apply_gradients(&mut self, learning_rate: f64) -> Result<(), String> {
        self.attention_norm.apply_gradients(learning_rate);
        self.attention.apply_gradients(learning_rate)?;
        self.feedforward_norm.apply_gradients(learning_rate);
        self.feedforward.apply_gradients(learning_rate);
        Ok(())
    }

    pub(crate) fn write_checkpoint_weights(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        self.attention_norm.write_checkpoint_weights(writer)?;
        self.attention.write_checkpoint_weights(writer)?;
        self.feedforward_norm.write_checkpoint_weights(writer)?;
        self.feedforward.write_checkpoint_weights(writer)
    }

    pub(crate) fn read_checkpoint_weights(&mut self, reader: &mut dyn Read) -> Result<(), String> {
        self.attention_norm.read_checkpoint_weights(reader)?;
        self.attention.read_checkpoint_weights(reader)?;
        self.feedforward_norm.read_checkpoint_weights(reader)?;
        self.feedforward.read_checkpoint_weights(reader)
    }

    /// Checks the full block's input gradient end to end. MultiHeadAttention's
    /// W_O is private and has no perturbation API, so its direct finite-difference
    /// check remains in `multihead::tests`; this verifies the complete block chain.
    pub fn check_input_gradients(embed_dim: usize, num_heads: usize, ff_dim: usize, seq_len: usize, seed: u64) -> (bool, f64) {
        if embed_dim == 0 || seq_len == 0 { return (false, f64::INFINITY); }
        let mut block = match Self::new(embed_dim, num_heads, ff_dim, seed) { Ok(v) => v, Err(_) => return (false, f64::INFINITY) };
        let x = sample_matrix(seq_len, embed_dim, seed.wrapping_add(29));
        let (output, cache) = match block.forward_with_cache(&x) { Ok(v) => v, Err(_) => return (false, f64::INFINITY) };
        let upstream = vec![vec![1.0; embed_dim]; seq_len];
        let analytic = match block.backward(&x, &cache, &upstream) { Ok(v) => v, Err(_) => return (false, f64::INFINITY) };
        let _ = output;
        let eps = 1e-5;
        let mut max_error: f64 = 0.0;
        for idx in 0..8.min(seq_len * embed_dim) {
            let r = idx / embed_dim; let c = idx % embed_dim;
            let mut plus_x = x.clone(); plus_x[r][c] += eps;
            let mut minus_x = x.clone(); minus_x[r][c] -= eps;
            let plus = match block.forward(&plus_x) { Ok(v) => sum(&v), Err(_) => return (false, f64::INFINITY) };
            let minus = match block.forward(&minus_x) { Ok(v) => sum(&v), Err(_) => return (false, f64::INFINITY) };
            let numerical = (plus - minus) / (2.0 * eps);
            let relative = (numerical - analytic[r][c]).abs() / (1e-8 + numerical.abs().max(analytic[r][c].abs()));
            max_error = max_error.max(relative);
        }
        (max_error < 1e-3, max_error)
    }

    #[cfg(test)]
    pub(crate) fn test_parameter_value_and_gradient(&self, kind: usize, row: usize, column: usize) -> (f64, f64) {
        match kind {
            0..=5 => self.attention.test_parameter_value_and_gradient(Some(kind / 3), kind % 3, row, column),
            6 => self.attention.test_parameter_value_and_gradient(None, 3, row, column),
            7 => self.feedforward.test_parameter_value_and_gradient(0, row, column),
            8 => self.feedforward.test_parameter_value_and_gradient(1, row, column),
            9 => self.feedforward.test_parameter_value_and_gradient(2, row, column),
            10 => self.feedforward.test_parameter_value_and_gradient(3, row, column),
            11 => self.attention_norm.test_parameter_value_and_gradient(0, row),
            12 => self.attention_norm.test_parameter_value_and_gradient(1, row),
            13 => self.feedforward_norm.test_parameter_value_and_gradient(0, row),
            14 => self.feedforward_norm.test_parameter_value_and_gradient(1, row),
            _ => panic!("unknown TransformerBlock parameter kind"),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_perturb_parameter(&mut self, kind: usize, row: usize, column: usize, delta: f64) {
        match kind {
            0..=5 => self.attention.test_perturb_parameter(Some(kind / 3), kind % 3, row, column, delta),
            6 => self.attention.test_perturb_parameter(None, 3, row, column, delta),
            7 => self.feedforward.test_perturb_parameter(0, row, column, delta),
            8 => self.feedforward.test_perturb_parameter(1, row, column, delta),
            9 => self.feedforward.test_perturb_parameter(2, row, column, delta),
            10 => self.feedforward.test_perturb_parameter(3, row, column, delta),
            11 => self.attention_norm.test_perturb_parameter(0, row, delta),
            12 => self.attention_norm.test_perturb_parameter(1, row, delta),
            13 => self.feedforward_norm.test_perturb_parameter(0, row, delta),
            14 => self.feedforward_norm.test_perturb_parameter(1, row, delta),
            _ => panic!("unknown TransformerBlock parameter kind"),
        }
    }
}

fn validate(x: &[Vec<f64>], width: usize, name: &str) -> Result<(), String> {
    if x.iter().any(|row| row.len() != width) { return Err(format!("{name} rows must have width {width}")); }
    Ok(())
}
fn add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Matrix, String> {
    if a.len() != b.len() || a.iter().zip(b).any(|(x, y)| x.len() != y.len()) { return Err("residual operands must have matching shapes".into()); }
    Ok(a.iter().zip(b).map(|(x, y)| x.iter().zip(y).map(|(u, v)| u + v).collect()).collect())
}
fn sample_matrix(m: usize, n: usize, seed: u64) -> Matrix { (0..m).map(|r| (0..n).map(|c| (((r * 37 + c * 19 + seed as usize % 53) % 101) as f64 - 50.0) / 17.0).collect()).collect() }
fn sum(x: &[Vec<f64>]) -> f64 { x.iter().flatten().sum() }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn transformer_block_shape() {
        let block = TransformerBlock::new(64, 4, 256, 7).unwrap();
        let x = vec![vec![0.1; 64]; 5]; let output = block.forward(&x).unwrap();
        assert_eq!(output.len(), 5); assert_eq!(output[0].len(), 64);
    }
    #[test] fn full_block_input_gradient_check() {
        let (pass, error) = TransformerBlock::check_input_gradients(8, 2, 16, 3, 7);
        assert!(pass, "max relative error {error}");
        println!("TransformerBlock full-path input max relative gradient error: {error:.3e}");
    }
}
