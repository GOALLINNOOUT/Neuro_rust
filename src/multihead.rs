use std::io::{Read, Write};

use crate::attention::{AttentionCache, AttentionHead};
use crate::matmul::matmul as fast_matmul;

pub type Matrix = Vec<Vec<f64>>;

#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    heads: Vec<AttentionHead>,
    w_o: Matrix,
    grad_w_o: Matrix,
    embed_dim: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    pub(crate) fn diagnostic_extrema(&self) -> Vec<(&'static str, f64, f64)> {
        let mut out = vec![("W_O", max_abs(self.w_o.iter().flatten().copied()), max_abs(self.grad_w_o.iter().flatten().copied()))];
        for (index, head) in self.heads.iter().enumerate() {
            for (name, weight, grad) in head.diagnostic_extrema() {
                let label = match name { "Q" => "Q", "K" => "K", _ => "V" };
                out.push((if index == 0 { label } else { match label { "Q" => "Q", "K" => "K", _ => "V" } }, weight, grad));
            }
        }
        out
    }
    pub fn new(embed_dim: usize, num_heads: usize, seed: u64) -> Result<Self, String> {
        if embed_dim == 0 || num_heads == 0 {
            return Err("embed_dim and num_heads must be greater than zero".into());
        }
        if !embed_dim.is_multiple_of(num_heads) {
            return Err("embed_dim must be divisible by num_heads".into());
        }
        let head_dim = embed_dim / num_heads;
        // Each head gets a distinct deterministic seed so its projections differ.
        let mut heads = Vec::with_capacity(num_heads);
        for i in 0..num_heads {
            heads.push(AttentionHead::new(embed_dim, head_dim, seed.wrapping_add(i as u64))?);
        }
        let mut rng = SplitMix64::new(seed.wrapping_add(num_heads as u64));
        let w_o = random_matrix(embed_dim, embed_dim, &mut rng);
        Ok(Self { heads, w_o, grad_w_o: zeros(embed_dim, embed_dim), embed_dim, head_dim })
    }

    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Matrix, String> {
        self.forward_with_cache(x).map(|v| v.0)
    }

    pub fn forward_with_cache(&self, x: &[Vec<f64>]) -> Result<(Matrix, Vec<AttentionCache>, Matrix), String> {
        let mut caches = Vec::with_capacity(self.heads.len());
        let mut concat = vec![Vec::with_capacity(self.embed_dim); x.len()];
        for head in &self.heads {
            let cache = head.forward_with_cache(x)?;
            for (row, out) in concat.iter_mut().zip(&cache.0) { row.extend_from_slice(out); }
            caches.push(cache);
        }
        let output = fast_matmul(&concat, &self.w_o);
        Ok((output, caches, concat))
    }

    pub fn backward(&mut self, x: &[Vec<f64>], caches: &[AttentionCache], concat_heads: &[Vec<f64>], d_output: &[Vec<f64>]) -> Result<Matrix, String> {
        validate(x, self.embed_dim, "input")?;
        validate(concat_heads, self.embed_dim, "concatenated heads")?;
        validate(d_output, self.embed_dim, "output gradient")?;
        if caches.len() != self.heads.len() || concat_heads.len() != d_output.len() || x.len() != d_output.len() {
            return Err("sequence lengths and cache count must match the multi-head configuration".into());
        }
        let concat_t = transpose(concat_heads);
        let grad = fast_matmul(&concat_t, d_output);
        add_in_place(&mut self.grad_w_o, &grad);
        let d_concat = fast_matmul(d_output, &transpose(&self.w_o));
        let mut total_dx = zeros(x.len(), self.embed_dim);
        for (h, (head, cache)) in self.heads.iter_mut().zip(caches).enumerate() {
            let start = h * self.head_dim;
            let d_head: Matrix = d_concat.iter().map(|row| row[start..start + self.head_dim].to_vec()).collect();
            let dx = head.backward(x, &cache.1, &cache.2, &cache.3, &cache.4, &d_head)?;
            add_in_place(&mut total_dx, &dx);
        }
        Ok(total_dx)
    }

    pub fn apply_gradients(&mut self, learning_rate: f64) -> Result<(), String> {
        if !learning_rate.is_finite() || learning_rate < 0.0 { return Err("learning_rate must be finite and non-negative".into()); }
        for (w, g) in self.w_o.iter_mut().flatten().zip(self.grad_w_o.iter_mut().flatten()) { *w -= learning_rate * *g; *g = 0.0; }
        for head in &mut self.heads { head.apply_gradients(learning_rate)?; }
        Ok(())
    }

    pub(crate) fn write_checkpoint_weights(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        crate::checkpoint::write_matrix(writer, &self.w_o)?;
        for head in &self.heads {
            head.write_checkpoint_weights(writer)?;
        }
        Ok(())
    }

    pub(crate) fn read_checkpoint_weights(&mut self, reader: &mut dyn Read) -> Result<(), String> {
        self.w_o = crate::checkpoint::read_matrix(reader, self.embed_dim, self.embed_dim)
            .map_err(|e| e.to_string())?;
        for head in &mut self.heads {
            head.read_checkpoint_weights(reader)?;
        }
        self.grad_w_o = zeros(self.embed_dim, self.embed_dim);
        Ok(())
    }

    pub fn check_gradients(embed_dim: usize, num_heads: usize, seq_len: usize, seed: u64) -> (bool, f64) {
        let Ok(mut model) = Self::new(embed_dim, num_heads, seed) else { return (false, f64::INFINITY) };
        if seq_len == 0 { return (false, f64::INFINITY); }
        let mut rng = SplitMix64::new(seed.wrapping_add(991));
        let x = random_matrix(seq_len, embed_dim, &mut rng);
        let (output, caches, concat) = match model.forward_with_cache(&x) { Ok(v) => v, Err(_) => return (false, f64::INFINITY) };
        let d_output = vec![vec![1.0; embed_dim]; seq_len];
        if model.backward(&x, &caches, &concat, &d_output).is_err() { return (false, f64::INFINITY); }
        let analytic = model.grad_w_o.clone();
        let eps = 1e-5;
        let mut max_error: f64 = 0.0;
        for idx in 0..8.min(embed_dim * embed_dim) {
            let r = idx / embed_dim; let c = idx % embed_dim;
            model.w_o[r][c] += eps;
            let plus = model.forward(&x).map(|m| sum(&m)).unwrap_or(f64::NAN);
            model.w_o[r][c] -= 2.0 * eps;
            let minus = model.forward(&x).map(|m| sum(&m)).unwrap_or(f64::NAN);
            model.w_o[r][c] += eps;
            let numerical = (plus - minus) / (2.0 * eps);
            let rel = (numerical - analytic[r][c]).abs() / (1e-8 + numerical.abs().max(analytic[r][c].abs()));
            max_error = max_error.max(rel);
        }
        let _ = output;
        (max_error < 1e-4, max_error)
    }

    #[cfg(test)]
    pub(crate) fn test_parameter_value_and_gradient(&self, head: Option<usize>, kind: usize, row: usize, column: usize) -> (f64, f64) {
        match head {
            Some(index) => self.heads[index].test_parameter_value_and_gradient(kind, row, column),
            None if kind == 3 => (self.w_o[row][column], self.grad_w_o[row][column]),
            None => panic!("output projection uses parameter kind 3"),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_perturb_parameter(&mut self, head: Option<usize>, kind: usize, row: usize, column: usize, delta: f64) {
        match head {
            Some(index) => self.heads[index].test_perturb_parameter(kind, row, column, delta),
            None if kind == 3 => self.w_o[row][column] += delta,
            None => panic!("output projection uses parameter kind 3"),
        }
    }
}

fn max_abs(values: impl Iterator<Item = f64>) -> f64 { values.map(f64::abs).fold(0.0, f64::max) }

fn validate(a: &[Vec<f64>], width: usize, name: &str) -> Result<(), String> {
    if a.iter().any(|r| r.len() != width) { Err(format!("{name} rows must have width {width}")) } else { Ok(()) }
}
#[allow(dead_code)] // Retained temporarily for the scalar/SIMD equivalence check.
fn matmul_scalar_legacy(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Matrix, String> {
    if a.is_empty() { return Ok(Vec::new()); }
    let k = a[0].len();
    if a.iter().any(|r| r.len() != k) || b.len() != k { return Err("incompatible matrix dimensions".into()); }
    let n = b.first().map_or(0, Vec::len);
    if b.iter().any(|r| r.len() != n) { return Err("ragged right-hand matrix".into()); }
    let mut out = zeros(a.len(), n);
    for i in 0..a.len() { for p in 0..k { for j in 0..n { out[i][j] += a[i][p] * b[p][j]; } } }
    Ok(out)
}
fn transpose(a: &[Vec<f64>]) -> Matrix { if a.is_empty() { return Vec::new(); } (0..a[0].len()).map(|j| a.iter().map(|r| r[j]).collect()).collect() }
fn zeros(m: usize, n: usize) -> Matrix { vec![vec![0.0; n]; m] }
fn add_in_place(a: &mut Matrix, b: &[Vec<f64>]) { for (ar, br) in a.iter_mut().zip(b) { for (x, y) in ar.iter_mut().zip(br) { *x += y; } } }
fn sum(a: &[Vec<f64>]) -> f64 { a.iter().flatten().sum() }
fn random_matrix(m: usize, n: usize, rng: &mut SplitMix64) -> Matrix { (0..m).map(|_| (0..n).map(|_| (rng.next_f64() * 2.0 - 1.0) * 0.05).collect()).collect() }
struct SplitMix64(u64);
impl SplitMix64 { fn new(seed: u64) -> Self { Self(seed) } fn next_f64(&mut self) -> f64 { self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15); let mut z = self.0; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); z ^= z >> 31; (z >> 11) as f64 / ((1u64 << 53) as f64) } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn shape_and_dimension_validation() { let m = MultiHeadAttention::new(64, 4, 7).unwrap(); let x = vec![vec![0.1; 64]; 5]; assert_eq!(m.forward(&x).unwrap().len(), 5); assert_eq!(m.forward(&x).unwrap()[0].len(), 64); assert!(MultiHeadAttention::new(10, 3, 1).is_err()); }
    #[test] fn output_projection_gradient_check() { let (pass, error) = MultiHeadAttention::check_gradients(8, 2, 3, 4); assert!(pass, "max relative error {error}"); println!("Multi-head W_O max relative gradient error: {error:.3e}"); }
    #[test] fn update_reduces_sum_loss() { let mut m = MultiHeadAttention::new(4, 2, 8).unwrap(); let x = vec![vec![0.2; 4]; 3]; let (y, c, concat) = m.forward_with_cache(&x).unwrap(); let before = sum(&y); m.backward(&x, &c, &concat, &vec![vec![1.0; 4]; 3]).unwrap(); m.apply_gradients(0.01).unwrap(); let after = sum(&m.forward(&x).unwrap()); assert!(after < before, "{before} -> {after}"); }
}
