/// Fixed sinusoidal position vectors that are added to token embeddings.
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    table: Vec<Vec<f64>>,
    embed_dim: usize,
}

impl PositionalEncoding {
    /// Precomputes the standard sinusoidal table for every position.
    ///
    /// Different frequencies make nearby positions have similar encodings
    /// (smooth local context), while distant positions diverge. This gives the
    /// network a representation from which it can learn relative distance, not
    /// just memorize an absolute position.
    pub fn new(max_seq_len: usize, embed_dim: usize) -> Result<Self, String> {
        if !embed_dim.is_multiple_of(2) {
            return Err(format!(
                "embedding dimension must be even for paired sine/cosine encodings; got {embed_dim}"
            ));
        }

        let mut table = Vec::with_capacity(max_seq_len);
        for position in 0..max_seq_len {
            let mut encoding = Vec::with_capacity(embed_dim);
            for dimension in 0..embed_dim {
                let i = dimension / 2;
                let denominator = 10_000_f64.powf((2 * i) as f64 / embed_dim as f64);
                let angle = position as f64 / denominator;
                if dimension.is_multiple_of(2) {
                    encoding.push(angle.sin());
                } else {
                    encoding.push(angle.cos());
                }
            }
            table.push(encoding);
        }

        Ok(Self { table, embed_dim })
    }

    /// Returns the precomputed encoding for one zero-based sequence position.
    pub fn encode(&self, position: usize) -> Result<&[f64], String> {
        self.table.get(position).map(Vec::as_slice).ok_or_else(|| {
            format!(
                "position {position} is out of range for positional table with maximum sequence length {}",
                self.table.len()
            )
        })
    }

    /// Adds positional vectors to embeddings and returns a newly allocated
    /// sequence. Inputs and the precomputed table are left unchanged.
    pub fn apply(&self, embeddings: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if embeddings.len() > self.table.len() {
            return Err(format!(
                "embedding sequence length ({}) exceeds positional table maximum ({})",
                embeddings.len(),
                self.table.len()
            ));
        }

        let mut output = Vec::with_capacity(embeddings.len());
        for (position, embedding) in embeddings.iter().enumerate() {
            if embedding.len() != self.embed_dim {
                return Err(format!(
                    "embedding at position {position} has dimension {}, expected {}",
                    embedding.len(),
                    self.embed_dim
                ));
            }
            let encoding = &self.table[position];
            output.push(
                embedding
                    .iter()
                    .zip(encoding)
                    .map(|(value, position_value)| value + position_value)
                    .collect(),
            );
        }
        Ok(output)
    }

    pub fn max_seq_len(&self) -> usize {
        self.table.len()
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

#[cfg(test)]
mod tests {
    use super::PositionalEncoding;

    #[test]
    fn precomputes_expected_table_shape_and_checks_bounds() {
        let encoding = PositionalEncoding::new(10, 64).unwrap();
        assert_eq!(encoding.table.len(), 10);
        assert_eq!(encoding.encode(0).unwrap().len(), 64);
        assert_eq!(encoding.encode(9).unwrap().len(), 64);
        assert_eq!(
            encoding.encode(10).unwrap_err(),
            "position 10 is out of range for positional table with maximum sequence length 10"
        );
    }

    #[test]
    fn sinusoidal_values_match_hand_calculated_positions() {
        let encoding = PositionalEncoding::new(2, 4).unwrap();
        // At p=0 both angles are zero: sin(0)=0 and cos(0)=1 exactly.
        assert_eq!(encoding.encode(0).unwrap()[0], 0.0);
        assert_eq!(encoding.encode(0).unwrap()[1], 1.0);
        // At p=1,d=0 the denominator is 10000^0=1, so sin(1)=0.8414709848...
        assert!((encoding.encode(1).unwrap()[0] - 0.8414709848078965).abs() < 1e-10);
    }

    #[test]
    fn odd_embedding_dimension_returns_a_clear_error() {
        assert_eq!(
            PositionalEncoding::new(10, 5).unwrap_err(),
            "embedding dimension must be even for paired sine/cosine encodings; got 5"
        );
    }

    #[test]
    fn apply_adds_the_encoding_to_each_zero_embedding() {
        let encoding = PositionalEncoding::new(10, 4).unwrap();
        let zero_embeddings = vec![vec![0.0; 4]; 3];
        let output = encoding.apply(&zero_embeddings).unwrap();
        assert_eq!(output.len(), 3);
        for (position, vector) in output.iter().enumerate() {
            assert_eq!(vector, encoding.encode(position).unwrap());
        }
    }

    #[test]
    fn apply_rejects_sequences_longer_than_the_table() {
        let encoding = PositionalEncoding::new(2, 4).unwrap();
        let embeddings = vec![vec![0.0; 4]; 3];
        assert_eq!(
            encoding.apply(&embeddings).unwrap_err(),
            "embedding sequence length (3) exceeds positional table maximum (2)"
        );
    }

    #[test]
    fn apply_rejects_embedding_dimension_mismatch() {
        let encoding = PositionalEncoding::new(3, 4).unwrap();
        assert_eq!(
            encoding.apply(&[vec![0.0; 3]]).unwrap_err(),
            "embedding at position 0 has dimension 3, expected 4"
        );
    }
}
