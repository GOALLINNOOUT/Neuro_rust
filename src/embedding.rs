use std::io::{Read, Write};

/// Small initial weights are sampled uniformly from [-0.05, 0.05). The
/// standard library has no public seeded RNG and this crate has no shared
/// random utility, so this module uses a compact deterministic SplitMix64
/// sequence to keep a seed reproducible without adding a dependency.
const INITIAL_WEIGHT_LIMIT: f64 = 0.05;
const DEFAULT_NUMERIC_SCALE: f64 = 100.0;

#[derive(Debug, Clone)]
pub struct EmbeddingLayer {
    weights: Vec<Vec<f64>>,
    gradients: Vec<Vec<f64>>,
    numeric_scale: f64,
}

impl EmbeddingLayer {
    pub(crate) fn diagnostic_extrema(&self) -> (f64, f64) {
        (max_abs(self.weights.iter().flatten().copied()), max_abs(self.gradients.iter().flatten().copied()))
    }
    pub fn new(vocab_size: usize, embed_dim: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let weights = (0..vocab_size)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| (rng.next_unit_f64() * 2.0 - 1.0) * INITIAL_WEIGHT_LIMIT)
                    .collect()
            })
            .collect();
        let gradients = vec![vec![0.0; embed_dim]; vocab_size];
        Self {
            weights,
            gradients,
            numeric_scale: DEFAULT_NUMERIC_SCALE,
        }
    }

    /// Returns a cloned embedding for each token, or an error for an invalid ID.
    pub fn forward(&self, token_ids: &[usize]) -> Result<Vec<Vec<f64>>, String> {
        token_ids
            .iter()
            .map(|&token_id| {
                self.weights.get(token_id).cloned().ok_or_else(|| {
                    format!(
                        "token id {token_id} is out of range for embedding vocabulary of size {}",
                        self.weights.len()
                    )
                })
            })
            .collect()
    }

    /// Adds each supplied numeric value, divided by `numeric_scale`, to every
    /// dimension of its corresponding token vector. Scaling keeps values such
    /// as 20.43 from overwhelming the small learned embedding weights; the
    /// tokenizer's parallel channel should supply `Some` only for `<NUM>` IDs.
    pub fn forward_with_numerics(
        &self,
        token_ids: &[usize],
        numeric_values: &[Option<f64>],
    ) -> Result<Vec<Vec<f64>>, String> {
        if token_ids.len() != numeric_values.len() {
            return Err(format!(
                "token_ids length ({}) must match numeric_values length ({})",
                token_ids.len(),
                numeric_values.len()
            ));
        }
        let mut output = self.forward(token_ids)?;
        for (embedding, numeric_value) in output.iter_mut().zip(numeric_values) {
            if let Some(value) = numeric_value {
                let addition = value / self.numeric_scale;
                for dimension in embedding {
                    *dimension += addition;
                }
            }
        }
        Ok(output)
    }

    /// Sets the positive finite numeric scale used by `forward_with_numerics`.
    pub fn set_numeric_scale(&mut self, scale: f64) -> Result<(), String> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err("numeric scale must be a positive finite number".to_string());
        }
        self.numeric_scale = scale;
        Ok(())
    }

    /// Adds one token's embedding gradient to its accumulator row.
    pub fn accumulate_grad(&mut self, token_id: usize, grad: &[f64]) -> Result<(), String> {
        if grad.len() != self.embedding_dim() {
            return Err(format!(
                "gradient length ({}) must match embedding dimension ({})",
                grad.len(),
                self.embedding_dim()
            ));
        }
        let gradient_row = self.gradients.get_mut(token_id).ok_or_else(|| {
            format!(
                "token id {token_id} is out of range for embedding vocabulary of size {}",
                self.weights.len()
            )
        })?;
        for (accumulated, value) in gradient_row.iter_mut().zip(grad) {
            *accumulated += value;
        }
        Ok(())
    }

    /// Applies an SGD update and clears all accumulated gradients.
    pub fn apply_gradients(&mut self, learning_rate: f64) -> Result<(), String> {
        if !learning_rate.is_finite() || learning_rate < 0.0 {
            return Err("learning rate must be a non-negative finite number".to_string());
        }
        for (weight_row, gradient_row) in self.weights.iter_mut().zip(&mut self.gradients) {
            for (weight, gradient) in weight_row.iter_mut().zip(gradient_row) {
                *weight -= learning_rate * *gradient;
                *gradient = 0.0;
            }
        }
        Ok(())
    }

    pub fn vocab_size(&self) -> usize {
        self.weights.len()
    }

    pub fn embedding_dim(&self) -> usize {
        self.weights.first().map_or(0, Vec::len)
    }

    pub(crate) fn write_checkpoint_weights(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        crate::checkpoint::write_f64(writer, self.numeric_scale)?;
        crate::checkpoint::write_matrix(writer, &self.weights)
    }

    pub(crate) fn read_checkpoint_weights(&mut self, reader: &mut dyn Read) -> Result<(), String> {
        let numeric_scale = crate::checkpoint::read_f64(reader).map_err(|e| e.to_string())?;
        if numeric_scale <= 0.0 {
            return Err("checkpoint numeric scale must be positive".into());
        }
        let weights = crate::checkpoint::read_matrix(
            reader,
            self.vocab_size(),
            self.embedding_dim(),
        )
        .map_err(|e| e.to_string())?;
        self.weights = weights;
        self.gradients = vec![vec![0.0; self.embedding_dim()]; self.vocab_size()];
        self.numeric_scale = numeric_scale;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn test_parameter_value_and_gradient(&self, token: usize, dimension: usize) -> (f64, f64) {
        (self.weights[token][dimension], self.gradients[token][dimension])
    }

    #[cfg(test)]
    pub(crate) fn test_perturb_parameter(&mut self, token: usize, dimension: usize, delta: f64) {
        self.weights[token][dimension] += delta;
    }
}

fn max_abs(values: impl Iterator<Item = f64>) -> f64 { values.map(f64::abs).fold(0.0, f64::max) }

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn next_unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::EmbeddingLayer;

    #[test]
    fn forward_returns_seeded_rows_with_expected_shape() {
        let layer = EmbeddingLayer::new(5, 3, 42);
        let other_same_seed = EmbeddingLayer::new(5, 3, 42);
        let ids = [3, 1];
        let output = layer.forward(&ids).unwrap();
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|row| row.len() == 3));
        assert_eq!(output[0], layer.weights[3]);
        assert_eq!(output[1], layer.weights[1]);
        assert_eq!(layer.weights, other_same_seed.weights);
    }

    #[test]
    fn numeric_value_is_scaled_and_added_to_each_dimension() {
        let layer = EmbeddingLayer::new(8, 4, 7);
        let ids = [0, 1, 4, 3];
        let plain = layer.forward(&ids).unwrap();
        let with_numbers = layer
            .forward_with_numerics(&ids, &[None, None, Some(20.43), None])
            .unwrap();
        assert_eq!(with_numbers.len(), ids.len());
        assert!(with_numbers.iter().all(|row| row.len() == 4));
        for dimension in 0..4 {
            assert!((with_numbers[2][dimension] - plain[2][dimension] - 0.2043).abs() < 1e-12);
            assert_eq!(with_numbers[0][dimension], plain[0][dimension]);
            assert_eq!(with_numbers[1][dimension], plain[1][dimension]);
            assert_eq!(with_numbers[3][dimension], plain[3][dimension]);
        }
    }

    #[test]
    fn forward_rejects_out_of_range_token_ids_clearly() {
        let layer = EmbeddingLayer::new(3, 2, 1);
        assert_eq!(
            layer.forward(&[3]).unwrap_err(),
            "token id 3 is out of range for embedding vocabulary of size 3"
        );
    }

    #[test]
    fn gradient_accumulation_and_sgd_update_have_expected_direction_and_size() {
        let mut layer = EmbeddingLayer::new(2, 3, 11);
        let before = layer.weights[1].clone();
        let untouched_row = layer.weights[0].clone();
        let gradient = [1.0, -2.0, 0.5];
        let learning_rate = 0.1;
        layer.accumulate_grad(1, &gradient).unwrap();
        layer.apply_gradients(learning_rate).unwrap();
        for dimension in 0..3 {
            let change = layer.weights[1][dimension] - before[dimension];
            assert!((change + learning_rate * gradient[dimension]).abs() < 1e-12);
        }
        assert_eq!(layer.gradients[1], [0.0, 0.0, 0.0]);
        assert_eq!(layer.weights[0], untouched_row);
    }

    #[test]
    fn numeric_forward_rejects_mismatched_lengths() {
        let layer = EmbeddingLayer::new(3, 2, 1);
        assert_eq!(
            layer
                .forward_with_numerics(&[0, 1], &[Some(2.0)])
                .unwrap_err(),
            "token_ids length (2) must match numeric_values length (1)"
        );
    }

    #[test]
    fn numeric_scale_can_be_configured_and_must_be_positive_finite() {
        let mut layer = EmbeddingLayer::new(2, 2, 1);
        layer.set_numeric_scale(10.0).unwrap();
        assert_eq!(layer.numeric_scale, 10.0);
        assert!(layer.set_numeric_scale(0.0).is_err());
        assert!(layer.set_numeric_scale(f64::INFINITY).is_err());
    }
}
