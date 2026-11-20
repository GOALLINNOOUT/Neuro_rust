use std::io::{Read, Write};
use std::cell::Cell;
use crate::matmul::matmul as fast_matmul;

const INITIAL_WEIGHT_LIMIT: f64 = 0.05;
const CAUSAL_MASK_VALUE: f64 = -1.0e9;

thread_local! { static DIAG_MAX_SCORE: Cell<f64> = const { Cell::new(0.0) }; }
pub fn reset_diagnostic_score() { DIAG_MAX_SCORE.with(|v| v.set(0.0)); }
pub fn diagnostic_max_score() -> f64 { DIAG_MAX_SCORE.with(Cell::get) }

pub type Matrix = Vec<Vec<f64>>;
pub type AttentionCache = (Matrix, Matrix, Matrix, Matrix, Matrix);

/// One causal scaled-dot-product attention head.
#[derive(Debug, Clone)]
pub struct AttentionHead {
    w_q: Vec<Vec<f64>>,
    w_k: Vec<Vec<f64>>,
    w_v: Vec<Vec<f64>>,
    grad_w_q: Vec<Vec<f64>>,
    grad_w_k: Vec<Vec<f64>>,
    grad_w_v: Vec<Vec<f64>>,
    embed_dim: usize,
    head_dim: usize,
}

struct ForwardIntermediates {
    output: Vec<Vec<f64>>,
    queries: Vec<Vec<f64>>,
    keys: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
    _attention_weights: Vec<Vec<f64>>,
}

#[derive(Debug)]
struct GradientCheckFailure {
    row: usize,
    column: usize,
    numerical: f64,
    analytic: f64,
    relative_error: f64,
}

#[derive(Debug)]
struct GradientCheckReport {
    checked_weights: usize,
    max_relative_error: f64,
    failure: Option<GradientCheckFailure>,
}

impl AttentionHead {
    pub(crate) fn diagnostic_extrema(&self) -> Vec<(&'static str, f64, f64)> {
        vec![
            ("Q", max_abs(self.w_q.iter().flatten().copied()), max_abs(self.grad_w_q.iter().flatten().copied())),
            ("K", max_abs(self.w_k.iter().flatten().copied()), max_abs(self.grad_w_k.iter().flatten().copied())),
            ("V", max_abs(self.w_v.iter().flatten().copied()), max_abs(self.grad_w_v.iter().flatten().copied())),
        ]
    }
    pub fn new(embed_dim: usize, head_dim: usize, seed: u64) -> Result<Self, String> {
        if embed_dim == 0 {
            return Err("embed_dim must be greater than zero".to_string());
        }
        if head_dim == 0 {
            return Err("head_dim must be greater than zero".to_string());
        }

        let mut rng = SplitMix64::new(seed);
        let w_q = random_matrix(embed_dim, head_dim, &mut rng);
        let w_k = random_matrix(embed_dim, head_dim, &mut rng);
        let w_v = random_matrix(embed_dim, head_dim, &mut rng);
        let grad_w_q = zero_matrix(embed_dim, head_dim);
        let grad_w_k = zero_matrix(embed_dim, head_dim);
        let grad_w_v = zero_matrix(embed_dim, head_dim);
        Ok(Self {
            w_q,
            w_k,
            w_v,
            grad_w_q,
            grad_w_k,
            grad_w_v,
            embed_dim,
            head_dim,
        })
    }

    /// Projects inputs into Q, K, and V, computes scaled causal attention, and
    /// returns one weighted value vector per input position.
    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        Ok(self.forward_intermediates(x)?.output)
    }

    /// Runs the forward pass and returns `(O, Q, K, V, A)` so callers can cache
    /// the values required by `backward` without re-deriving them.
    pub fn forward_with_cache(&self, x: &[Vec<f64>]) -> Result<AttentionCache, String> {
        let pass = self.forward_intermediates(x)?;
        Ok((
            pass.output,
            pass.queries,
            pass.keys,
            pass.values,
            pass._attention_weights,
        ))
    }

    fn forward_intermediates(&self, x: &[Vec<f64>]) -> Result<ForwardIntermediates, String> {
        validate_matrix_width(x, self.embed_dim, "input")?;

        // A high score between positions i and j means position i "decides"
        // that position j is relevant and should inform its output representation.
        let queries = fast_matmul(x, &self.w_q);
        let keys = fast_matmul(x, &self.w_k);
        let values = fast_matmul(x, &self.w_v);

        let transposed_keys = matrix_transpose(&keys);
        let raw_scores = fast_matmul(&queries, &transposed_keys);
        let mut scaled_scores = raw_scores;
        scale_scores(&mut scaled_scores, self.head_dim);

        // During generation tokens are produced left-to-right. This mask also
        // trains the head under that same constraint by hiding every future token.
        let mut masked_scores = scaled_scores;
        for (query_position, score_row) in masked_scores.iter_mut().enumerate() {
            for (key_position, score) in score_row.iter_mut().enumerate() {
                if key_position > query_position {
                    *score = CAUSAL_MASK_VALUE;
                }
            }
        }

        let attention_weights = softmax_rows(&masked_scores)?;
        let output = fast_matmul(&attention_weights, &values);
        Ok(ForwardIntermediates {
            output,
            queries,
            keys,
            values,
            _attention_weights: attention_weights,
        })
    }

    /// Backpropagates through output mixing, softmax, causal masking, scaling,
    /// and the Q/K projections. Projection-weight gradients accumulate across
    /// calls until `apply_gradients` is invoked.
    pub fn backward(
        &mut self,
        x: &[Vec<f64>],
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
        attn_weights: &[Vec<f64>],
        d_output: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, String> {
        let seq_len = x.len();
        validate_matrix_width(x, self.embed_dim, "input")?;
        validate_sequence_width(q, seq_len, self.head_dim, "queries")?;
        validate_sequence_width(k, seq_len, self.head_dim, "keys")?;
        validate_sequence_width(v, seq_len, self.head_dim, "values")?;
        validate_sequence_width(attn_weights, seq_len, seq_len, "attention weights")?;
        validate_sequence_width(d_output, seq_len, self.head_dim, "output gradient")?;
        if seq_len == 0 {
            return Ok(Vec::new());
        }

        // STEP 1: O = A·V, so differentiate both factors.
        let transposed_v = matrix_transpose(v);
        let d_a = fast_matmul(d_output, &transposed_v);
        let transposed_attention = matrix_transpose(attn_weights);
        let d_v = fast_matmul(&transposed_attention, d_output);
        let transposed_x = matrix_transpose(x);
        let grad_w_v_contribution = fast_matmul(&transposed_x, &d_v);
        add_matrix_in_place(&mut self.grad_w_v, &grad_w_v_contribution)?;
        let transposed_w_v = matrix_transpose(&self.w_v);
        let d_x_from_v = fast_matmul(&d_v, &transposed_w_v);

        // STEP 2: Softmax derivative derivation:
        // dL/dz_j = sum_k (dL/da_k * da_k/dz_j).
        // For softmax, da_k/dz_j = a_k * (delta_kj - a_j), so substituting gives
        // sum_k (g_k * a_k * delta_kj) - a_j * sum_k (g_k * a_k)
        // = g_j * a_j - a_j * sum_k (g_k * a_k)
        // = a_j * (g_j - sum_k (g_k * a_k)).
        let mut d_scores = vec![vec![0.0; seq_len]; seq_len];
        for row in 0..seq_len {
            let weighted_gradient: f64 = d_a[row]
                .iter()
                .zip(&attn_weights[row])
                .map(|(gradient, probability)| gradient * probability)
                .sum();
            for column in 0..seq_len {
                d_scores[row][column] =
                    attn_weights[row][column] * (d_a[row][column] - weighted_gradient);
            }
        }

        // STEP 3: masked future scores did not participate in attention.
        for (query_position, score_row) in d_scores.iter_mut().enumerate() {
            for (key_position, score) in score_row.iter_mut().enumerate() {
                if key_position > query_position {
                    *score = 0.0;
                }
            }
        }

        // STEP 4: the forward scaling factor is 1/sqrt(head_dim), so apply
        // that same factor to the gradient flowing to the unscaled scores.
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let d_scores_unscaled: Vec<Vec<f64>> = d_scores
            .iter()
            .map(|row| row.iter().map(|gradient| gradient * scale).collect())
            .collect();

        // STEP 5: scores = Q·K^T, then accumulate projection-weight gradients.
        let d_q = fast_matmul(&d_scores_unscaled, k);
        let transposed_d_scores_unscaled = matrix_transpose(&d_scores_unscaled);
        let d_k = fast_matmul(&transposed_d_scores_unscaled, q);
        let grad_w_q_contribution = fast_matmul(&transposed_x, &d_q);
        add_matrix_in_place(&mut self.grad_w_q, &grad_w_q_contribution)?;
        let grad_w_k_contribution = fast_matmul(&transposed_x, &d_k);
        add_matrix_in_place(&mut self.grad_w_k, &grad_w_k_contribution)?;

        let transposed_w_q = matrix_transpose(&self.w_q);
        let d_x_from_q = fast_matmul(&d_q, &transposed_w_q);
        let transposed_w_k = matrix_transpose(&self.w_k);
        let d_x_from_k = fast_matmul(&d_k, &transposed_w_k);
        let d_x_from_qk = add_matrices(&d_x_from_q, &d_x_from_k)?;

        // STEP 6: combine the value path and query/key paths into dX.
        let d_x = add_matrices(&d_x_from_v, &d_x_from_qk)?;
        Ok(d_x)
    }

    /// Applies an SGD update to all three projection matrices, then clears
    /// their gradient accumulators.
    pub fn apply_gradients(&mut self, learning_rate: f64) -> Result<(), String> {
        if !learning_rate.is_finite() || learning_rate < 0.0 {
            return Err("learning rate must be a non-negative finite number".to_string());
        }
        apply_matrix_gradients(&mut self.w_q, &mut self.grad_w_q, learning_rate);
        apply_matrix_gradients(&mut self.w_k, &mut self.grad_w_k, learning_rate);
        apply_matrix_gradients(&mut self.w_v, &mut self.grad_w_v, learning_rate);
        Ok(())
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub(crate) fn write_checkpoint_weights(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        crate::checkpoint::write_matrix(writer, &self.w_q)?;
        crate::checkpoint::write_matrix(writer, &self.w_k)?;
        crate::checkpoint::write_matrix(writer, &self.w_v)
    }

    pub(crate) fn read_checkpoint_weights(&mut self, reader: &mut dyn Read) -> Result<(), String> {
        self.w_q = crate::checkpoint::read_matrix(reader, self.embed_dim, self.head_dim)
            .map_err(|e| e.to_string())?;
        self.w_k = crate::checkpoint::read_matrix(reader, self.embed_dim, self.head_dim)
            .map_err(|e| e.to_string())?;
        self.w_v = crate::checkpoint::read_matrix(reader, self.embed_dim, self.head_dim)
            .map_err(|e| e.to_string())?;
        self.grad_w_q = zero_matrix(self.embed_dim, self.head_dim);
        self.grad_w_k = zero_matrix(self.embed_dim, self.head_dim);
        self.grad_w_v = zero_matrix(self.embed_dim, self.head_dim);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn test_parameter_value_and_gradient(&self, kind: usize, row: usize, column: usize) -> (f64, f64) {
        match kind {
            0 => (self.w_q[row][column], self.grad_w_q[row][column]),
            1 => (self.w_k[row][column], self.grad_w_k[row][column]),
            2 => (self.w_v[row][column], self.grad_w_v[row][column]),
            _ => panic!("unknown attention projection kind"),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_perturb_parameter(&mut self, kind: usize, row: usize, column: usize, delta: f64) {
        match kind {
            0 => self.w_q[row][column] += delta,
            1 => self.w_k[row][column] += delta,
            2 => self.w_v[row][column] += delta,
            _ => panic!("unknown attention projection kind"),
        }
    }
}

/// Spot-checks eight deterministic W_Q entries against centered finite
/// differences. Returns false for invalid parameters or any failed check.
pub fn check_gradients(embed_dim: usize, head_dim: usize, seq_len: usize, seed: u64) -> bool {
    match gradient_check_details(embed_dim, head_dim, seq_len, seed) {
        Ok(report) => {
            if let Some(failure) = report.failure {
                eprintln!(
                    "gradient mismatch at W_Q[{}, {}]: numerical={}, analytic={}, relative_error={}",
                    failure.row,
                    failure.column,
                    failure.numerical,
                    failure.analytic,
                    failure.relative_error
                );
                false
            } else {
                report.checked_weights == 8 && report.max_relative_error <= 1e-4
            }
        }
        Err(error) => {
            eprintln!("gradient check could not run: {error}");
            false
        }
    }
}

fn gradient_check_details(
    embed_dim: usize,
    head_dim: usize,
    seq_len: usize,
    seed: u64,
) -> Result<GradientCheckReport, String> {
    let total_weights = embed_dim
        .checked_mul(head_dim)
        .ok_or_else(|| "weight count overflows usize".to_string())?;
    if total_weights < 8 {
        return Err("gradient check requires at least eight W_Q weights".to_string());
    }
    if seq_len == 0 {
        return Err("gradient check requires a non-empty sequence".to_string());
    }

    let mut head = AttentionHead::new(embed_dim, head_dim, seed)?;
    let mut input_rng = SplitMix64::new(seed ^ 0xA5A5_5A5A_D3C1_B7E9);
    let x = random_matrix(seq_len, embed_dim, &mut input_rng);
    let (output, q, k, v, attention_weights) = head.forward_with_cache(&x)?;
    let base_loss: f64 = output.iter().flatten().sum();
    let d_output = vec![vec![1.0; head_dim]; seq_len];
    head.backward(&x, &q, &k, &v, &attention_weights, &d_output)?;

    let mut selected: Vec<usize> = (0..total_weights).collect();
    let mut selection_rng = SplitMix64::new(seed ^ 0x6C8E_9CF5_7093_2BD5);
    for index in 0..8 {
        let remaining = total_weights - index;
        let swap_index = index + (selection_rng.next_u64() as usize % remaining);
        selected.swap(index, swap_index);
    }

    let epsilon = 1e-5;
    let mut max_relative_error: f64 = 0.0;
    let mut failure = None;
    for flat_index in selected.into_iter().take(8) {
        let row = flat_index / head_dim;
        let column = flat_index % head_dim;
        let original = head.w_q[row][column];

        head.w_q[row][column] = original + epsilon;
        let loss_plus: f64 = head.forward(&x)?.iter().flatten().sum();
        head.w_q[row][column] = original - epsilon;
        let loss_minus: f64 = head.forward(&x)?.iter().flatten().sum();
        head.w_q[row][column] = original;

        let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
        let analytic = head.grad_w_q[row][column];
        let relative_error = if numerical.is_finite() && analytic.is_finite() {
            (numerical - analytic).abs() / (numerical.abs() + analytic.abs() + 1e-8)
        } else {
            f64::INFINITY
        };
        max_relative_error = max_relative_error.max(relative_error);
        if relative_error > 1e-4 && failure.is_none() {
            failure = Some(GradientCheckFailure {
                row,
                column,
                numerical,
                analytic,
                relative_error,
            });
        }
    }

    // Keep the scalar objective calculation explicit, as specified for the check.
    let _loss_at_unperturbed_weights = base_loss;
    Ok(GradientCheckReport {
        checked_weights: 8,
        max_relative_error,
        failure,
    })
}

fn validate_matrix_width(matrix: &[Vec<f64>], width: usize, name: &str) -> Result<(), String> {
    for (row_index, row) in matrix.iter().enumerate() {
        if row.len() != width {
            return Err(format!(
                "{name} row {row_index} has dimension {}, expected {width}",
                row.len()
            ));
        }
    }
    Ok(())
}

fn validate_sequence_width(
    matrix: &[Vec<f64>],
    expected_rows: usize,
    expected_width: usize,
    name: &str,
) -> Result<(), String> {
    if matrix.len() != expected_rows {
        return Err(format!(
            "{name} has {} rows, expected {expected_rows}",
            matrix.len()
        ));
    }
    validate_matrix_width(matrix, expected_width, name)
}

#[allow(dead_code)] // Retained for comparison benchmarks and scalar verification.
pub(crate) fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    if a.is_empty() {
        return Ok(Vec::new());
    }
    let inner_dim = a[0].len();
    validate_matrix_width(a, inner_dim, "left matrix")?;
    validate_matrix_width(b, b.first().map_or(0, Vec::len), "right matrix")?;
    if b.len() != inner_dim {
        return Err(format!(
            "matrix multiplication dimension mismatch: left inner dimension {inner_dim}, right row count {}",
            b.len()
        ));
    }
    let cols = b.first().map_or(0, Vec::len);
    let mut out = vec![vec![0.0_f64; cols]; a.len()];
    for i in 0..a.len() {
        for k in 0..inner_dim {
            let aik = a[i][k];
            for j in 0..cols {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    Ok(out)
}

fn matrix_transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    (0..matrix[0].len())
        .map(|column| matrix.iter().map(|row| row[column]).collect())
        .collect()
}

fn add_matrices(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    if a.len() != b.len() {
        return Err(format!(
            "cannot add matrices with row counts {} and {}",
            a.len(),
            b.len()
        ));
    }
    a.iter()
        .zip(b)
        .enumerate()
        .map(|(row, (a_row, b_row))| {
            if a_row.len() != b_row.len() {
                return Err(format!(
                    "cannot add matrix row {row} with widths {} and {}",
                    a_row.len(),
                    b_row.len()
                ));
            }
            Ok(a_row.iter().zip(b_row).map(|(a, b)| a + b).collect())
        })
        .collect()
}

fn add_matrix_in_place(target: &mut [Vec<f64>], contribution: &[Vec<f64>]) -> Result<(), String> {
    if target.len() != contribution.len() {
        return Err(format!(
            "gradient accumulator has {} rows, contribution has {}",
            target.len(),
            contribution.len()
        ));
    }
    for (row, (target_row, contribution_row)) in target.iter_mut().zip(contribution).enumerate() {
        if target_row.len() != contribution_row.len() {
            return Err(format!(
                "gradient accumulator row {row} has width {}, contribution has {}",
                target_row.len(),
                contribution_row.len()
            ));
        }
        for (target_value, contribution_value) in target_row.iter_mut().zip(contribution_row) {
            *target_value += contribution_value;
        }
    }
    Ok(())
}

fn scale_scores(scores: &mut [Vec<f64>], head_dim: usize) {
    let scale = (head_dim as f64).sqrt();
    for row in scores {
        for score in row {
            *score /= scale;
        }
    }
}

fn softmax_rows(scores: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    DIAG_MAX_SCORE.with(|current| {
        let batch_max = scores.iter().flatten().filter(|&&score| score != CAUSAL_MASK_VALUE)
            .map(|score| if score.is_finite() { score.abs() } else { f64::INFINITY }).fold(0.0, f64::max);
        current.set(current.get().max(batch_max));
    });
    scores
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            if row.is_empty() {
                return Ok(Vec::new());
            }
            let maximum = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exponentials: Vec<f64> = row.iter().map(|score| (score - maximum).exp()).collect();
            let sum: f64 = exponentials.iter().sum();
            if !sum.is_finite() || sum == 0.0 {
                return Err(format!(
                    "softmax row {row_index} cannot be normalized because its exponential sum is {sum}"
                ));
            }
            Ok(exponentials.iter().map(|value| value / sum).collect())
        })
        .collect()
}

fn max_abs(values: impl Iterator<Item = f64>) -> f64 { values.map(f64::abs).fold(0.0, f64::max) }

fn random_matrix(rows: usize, columns: usize, rng: &mut SplitMix64) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|_| {
            (0..columns)
                .map(|_| (rng.next_unit_f64() * 2.0 - 1.0) * INITIAL_WEIGHT_LIMIT)
                .collect()
        })
        .collect()
}

fn zero_matrix(rows: usize, columns: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; columns]; rows]
}

fn apply_matrix_gradients(
    weights: &mut [Vec<f64>],
    gradients: &mut [Vec<f64>],
    learning_rate: f64,
) {
    for (weight_row, gradient_row) in weights.iter_mut().zip(gradients) {
        for (weight, gradient) in weight_row.iter_mut().zip(gradient_row) {
            *weight -= learning_rate * *gradient;
            *gradient = 0.0;
        }
    }
}

/// Same deterministic SplitMix64 sequence used by the embedding initializer.
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
    use super::{
        AttentionHead, check_gradients, gradient_check_details, matrix_multiply, scale_scores,
        softmax_rows,
    };

    #[test]
    fn forward_has_sequence_by_head_dimension_shape() {
        let head = AttentionHead::new(8, 4, 17).unwrap();
        let input = vec![vec![0.25; 8]; 4];
        let output = head.forward(&input).unwrap();
        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|row| row.len() == 4));
    }

    #[test]
    fn causal_mask_removes_all_attention_from_position_zero_to_future_positions() {
        let head = AttentionHead::new(2, 2, 9).unwrap();
        let input = vec![vec![0.0; 2]; 3];
        let pass = head.forward_intermediates(&input).unwrap();
        assert_eq!(pass._attention_weights[0][0], 1.0);
        assert!(pass._attention_weights[0][1] < 1e-6);
        assert!(pass._attention_weights[0][2] < 1e-6);
        assert_eq!(pass.output.len(), 3);
    }

    #[test]
    fn standalone_softmax_matches_reference_row() {
        let actual = softmax_rows(&[vec![1.0, 2.0, 3.0]]).unwrap();
        let expected = [0.0900, 0.2447, 0.6652];
        for (actual, expected) in actual[0].iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-4);
        }
        assert!((actual[0].iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn scaled_scores_are_smaller_than_the_unscaled_dot_products() {
        // With head_dim=4 and q=k=[2,0,0,0], the raw dot product is 4;
        // dividing by sqrt(4)=2 produces the pre-softmax score 2.
        let queries = vec![vec![2.0, 0.0, 0.0, 0.0]];
        let keys_transposed = vec![vec![2.0], vec![0.0], vec![0.0], vec![0.0]];
        let raw_scores = matrix_multiply(&queries, &keys_transposed).unwrap();
        let mut scaled_scores = raw_scores.clone();
        scale_scores(&mut scaled_scores, 4);
        assert_eq!(raw_scores[0][0], 4.0);
        assert_eq!(scaled_scores[0][0], 2.0);
        assert!(scaled_scores[0][0] < raw_scores[0][0]);
    }

    #[test]
    fn constructor_and_forward_validate_dimensions() {
        assert_eq!(
            AttentionHead::new(0, 2, 1).unwrap_err().to_string(),
            "embed_dim must be greater than zero"
        );
        assert_eq!(
            AttentionHead::new(2, 0, 1).unwrap_err().to_string(),
            "head_dim must be greater than zero"
        );
        let head = AttentionHead::new(3, 2, 1).unwrap();
        assert_eq!(
            head.forward(&[vec![0.0; 2]]).unwrap_err(),
            "input row 0 has dimension 2, expected 3"
        );
    }

    #[test]
    fn backward_returns_input_gradient_with_input_shape() {
        let mut head = AttentionHead::new(8, 4, 17).unwrap();
        let input = vec![vec![0.25; 8]; 4];
        let (_, q, k, v, attention_weights) = head.forward_with_cache(&input).unwrap();
        let d_output = vec![vec![1.0; 4]; 4];
        let d_x = head
            .backward(&input, &q, &k, &v, &attention_weights, &d_output)
            .unwrap();
        assert_eq!(d_x.len(), 4);
        assert!(d_x.iter().all(|row| row.len() == 8));
    }

    #[test]
    fn accumulated_gradients_update_weights_in_loss_reducing_direction() {
        let mut head = AttentionHead::new(8, 4, 42).unwrap();
        let mut rng = super::SplitMix64::new(77);
        let input = super::random_matrix(5, 8, &mut rng);
        let (output, q, k, v, attention_weights) = head.forward_with_cache(&input).unwrap();
        let loss_before: f64 = output.iter().flatten().sum();
        let d_output = vec![vec![1.0; 4]; 5];

        head.backward(&input, &q, &k, &v, &attention_weights, &d_output)
            .unwrap();
        head.backward(&input, &q, &k, &v, &attention_weights, &d_output)
            .unwrap();
        head.apply_gradients(0.1).unwrap();

        let loss_after: f64 = head.forward(&input).unwrap().iter().flatten().sum();
        assert!(
            loss_after < loss_before,
            "loss {loss_after} did not fall below {loss_before}"
        );
    }

    #[test]
    fn numerical_gradient_check_passes_for_eight_wq_weights() {
        let report = gradient_check_details(8, 4, 5, 42).unwrap();
        if let Some(failure) = &report.failure {
            panic!(
                "gradient mismatch at W_Q[{}, {}]: numerical={}, analytic={}, relative_error={}",
                failure.row,
                failure.column,
                failure.numerical,
                failure.analytic,
                failure.relative_error
            );
        }
        println!(
            "Gradient check passed: {} W_Q weights, maximum relative error {:.3e}",
            report.checked_weights, report.max_relative_error
        );
        // Failure indicates a chain-rule/backward-pass bug, not random test noise.
        assert!(check_gradients(8, 4, 5, 42));
    }
}
