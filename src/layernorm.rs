use std::io::{Read, Write};

pub type Matrix = Vec<Vec<f64>>;

#[derive(Debug, Clone)]
pub struct LayerNorm {
    gamma: Vec<f64>,
    beta: Vec<f64>,
    grad_gamma: Vec<f64>,
    grad_beta: Vec<f64>,
    epsilon: f64,
}

impl LayerNorm {
    pub(crate) fn diagnostic_extrema(&self) -> (f64, f64) {
        (max_abs(self.gamma.iter().chain(&self.beta).copied()), max_abs(self.grad_gamma.iter().chain(&self.grad_beta).copied()))
    }
    pub fn new(d_model: usize) -> Self {
        Self { gamma: vec![1.0; d_model], beta: vec![0.0; d_model], grad_gamma: vec![0.0; d_model], grad_beta: vec![0.0; d_model], epsilon: 1e-5 }
    }

    pub fn normalize(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.gamma.len(), "layer norm input width must match d_model");
        let x_norm = normalize_raw(x, self.epsilon);
        x_norm.iter().enumerate().map(|(i, v)| self.gamma[i] * v + self.beta[i]).collect()
    }

    pub fn forward(&self, x: &[Vec<f64>]) -> (Matrix, Matrix) {
        validate(x, self.gamma.len());
        let x_norm: Matrix = x.iter().map(|row| normalize_raw(row, self.epsilon)).collect();
        let output = x_norm.iter().map(|row| row.iter().enumerate().map(|(i, v)| self.gamma[i] * v + self.beta[i]).collect()).collect();
        (output, x_norm)
    }

    /// Backpropagates through affine scale/shift and normalization. `x` is
    /// needed to recover each row's centered values and inverse standard deviation.
    pub fn backward(&mut self, x: &[Vec<f64>], x_norm: &[Vec<f64>], d_output: &[Vec<f64>]) -> Matrix {
        let width = self.gamma.len();
        validate(x, width); validate(x_norm, width); validate(d_output, width);
        assert_eq!(x.len(), x_norm.len(), "input and normalized sequence lengths must match");
        assert_eq!(x.len(), d_output.len(), "input and gradient sequence lengths must match");
        let mut dx = vec![vec![0.0; width]; x.len()];
        for pos in 0..x.len() {
            for i in 0..width {
                self.grad_gamma[i] += d_output[pos][i] * x_norm[pos][i];
                self.grad_beta[i] += d_output[pos][i];
            }
            if width == 0 { continue; }
            let mean = x[pos].iter().sum::<f64>() / width as f64;
            let centered: Vec<f64> = x[pos].iter().map(|v| v - mean).collect();
            let variance = centered.iter().map(|v| v * v).sum::<f64>() / width as f64;
            let var_eps = variance + self.epsilon;
            let inv_std = var_eps.sqrt().recip();
            let d_x_norm: Vec<f64> = d_output[pos].iter().enumerate().map(|(i, g)| g * self.gamma[i]).collect();

            // From var_eps = mean((x-mean)^2)+epsilon: dvar sums
            // d_xnorm*(x-mean)*(-1/2)*var_eps^(-3/2). Then dmean includes
            // both the direct centered-input path and the variance path.
            let d_var = d_x_norm.iter().zip(&centered).map(|(g, c)| g * c * -0.5 * var_eps.powf(-1.5)).sum::<f64>();
            let d_mean = d_x_norm.iter().map(|g| -g * inv_std).sum::<f64>()
                + d_var * centered.iter().map(|c| -2.0 * c / width as f64).sum::<f64>();
            for i in 0..width {
                dx[pos][i] = d_x_norm[i] * inv_std + d_var * 2.0 * centered[i] / width as f64 + d_mean / width as f64;
            }
        }
        dx
    }

    pub fn apply_gradients(&mut self, learning_rate: f64) {
        for i in 0..self.gamma.len() { self.gamma[i] -= learning_rate * self.grad_gamma[i]; self.beta[i] -= learning_rate * self.grad_beta[i]; self.grad_gamma[i] = 0.0; self.grad_beta[i] = 0.0; }
    }

    pub(crate) fn write_checkpoint_weights(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        crate::checkpoint::write_f64s(writer, &self.gamma)?;
        crate::checkpoint::write_f64s(writer, &self.beta)
    }

    pub(crate) fn read_checkpoint_weights(&mut self, reader: &mut dyn Read) -> Result<(), String> {
        self.gamma = crate::checkpoint::read_f64s(reader, self.gamma.len())
            .map_err(|e| e.to_string())?;
        self.beta = crate::checkpoint::read_f64s(reader, self.beta.len())
            .map_err(|e| e.to_string())?;
        self.grad_gamma = vec![0.0; self.gamma.len()];
        self.grad_beta = vec![0.0; self.beta.len()];
        Ok(())
    }

    pub fn check_gamma_gradients(d_model: usize, seq_len: usize, seed: u64) -> (bool, f64) {
        if d_model == 0 || seq_len == 0 { return (false, f64::INFINITY); }
        let x = sample_matrix(seq_len, d_model, seed);
        let mut model = Self::new(d_model);
        let (_, x_norm) = model.forward(&x);
        let d_output = vec![vec![1.0; d_model]; seq_len];
        model.backward(&x, &x_norm, &d_output);
        let analytic = model.grad_gamma.clone();
        let mut max_error: f64 = 0.0;
        for (i, &analytic_value) in analytic.iter().enumerate().take(8.min(d_model)) {
            model.gamma[i] += 1e-5; let plus = sum(&model.forward(&x).0);
            model.gamma[i] -= 2e-5; let minus = sum(&model.forward(&x).0);
            model.gamma[i] += 1e-5;
            let numerical = (plus - minus) / 2e-5;
            let relative = (numerical - analytic_value).abs() / (1e-8 + numerical.abs().max(analytic_value.abs()));
            max_error = max_error.max(relative);
        }
        (max_error < 1e-4, max_error)
    }

    #[cfg(test)]
    pub(crate) fn test_parameter_value_and_gradient(&self, kind: usize, index: usize) -> (f64, f64) {
        match kind {
            0 => (self.gamma[index], self.grad_gamma[index]),
            1 => (self.beta[index], self.grad_beta[index]),
            _ => panic!("unknown LayerNorm parameter kind"),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_perturb_parameter(&mut self, kind: usize, index: usize, delta: f64) {
        match kind {
            0 => self.gamma[index] += delta,
            1 => self.beta[index] += delta,
            _ => panic!("unknown LayerNorm parameter kind"),
        }
    }
}
fn max_abs(values: impl Iterator<Item = f64>) -> f64 { values.map(f64::abs).fold(0.0, f64::max) }

fn normalize_raw(x: &[f64], epsilon: f64) -> Vec<f64> {
    if x.is_empty() { return Vec::new(); }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
    let inv_std = (variance + epsilon).sqrt().recip();
    x.iter().map(|v| (v - mean) * inv_std).collect()
}
fn validate(x: &[Vec<f64>], width: usize) { assert!(x.iter().all(|r| r.len() == width), "layer norm rows must have width {width}"); }
fn sample_matrix(m: usize, n: usize, seed: u64) -> Matrix { (0..m).map(|r| (0..n).map(|c| (((r * 37 + c * 19 + seed as usize % 53) % 101) as f64 - 50.0) / 17.0).collect()).collect() }
fn sum(x: &[Vec<f64>]) -> f64 { x.iter().flatten().sum() }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn known_vector_normalizes_to_zero_mean_and_unit_variance() {
        let x = [1.0, 2.0, 3.0, 4.0]; let norm = normalize_raw(&x, 0.0);
        let mean = norm.iter().sum::<f64>() / norm.len() as f64;
        let variance = norm.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / norm.len() as f64;
        assert!(mean.abs() < 1e-10); assert!((variance - 1.0).abs() < 1e-10);
    }
    #[test] fn initialized_affine_parameters_preserve_normalization() {
        let ln = LayerNorm::new(4); let x = [1.0, 2.0, 3.0, 4.0];
        let actual = ln.normalize(&x); let expected = normalize_raw(&x, 1e-5);
        for (a, e) in actual.iter().zip(expected) { assert!((a - e).abs() < 1e-12); }
    }
    #[test] fn gamma_numerical_gradient_check() { let (pass, error) = LayerNorm::check_gamma_gradients(8, 4, 7); assert!(pass, "max relative error {error}"); println!("LayerNorm gamma max relative gradient error: {error:.3e}"); }
}
