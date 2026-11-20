use std::collections::HashMap;

const SPECIAL_TOKENS: [&str; 5] = ["<BOS>", "<EOS>", "<PAD>", "<UNK>", "<NUM>"];
const BOS: usize = 0;
const EOS: usize = 1;
const PAD: usize = 2;
const UNK: usize = 3;
const NUM: usize = 4;

/// A deterministic word-level vocabulary with a separate channel for numeric
/// evidence. `max_vocab` includes the five mandatory special tokens; if a
/// smaller value is requested, the special tokens take precedence.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    word_to_id: HashMap<String, usize>,
    id_to_word: Vec<String>,
    configured_max_vocab: usize,
    corpus_unique_words: usize,
}

impl Tokenizer {
    pub const EOS_ID: usize = EOS;

    /// Restores a tokenizer from a checkpointed vocabulary without changing token IDs.
    pub(crate) fn from_checkpoint(
        id_to_word: Vec<String>,
        configured_max_vocab: usize,
        corpus_unique_words: usize,
    ) -> Result<Self, String> {
        if id_to_word.len() < SPECIAL_TOKENS.len()
            || id_to_word
                .iter()
                .take(SPECIAL_TOKENS.len())
                .map(String::as_str)
                .ne(SPECIAL_TOKENS)
        {
            return Err("checkpoint tokenizer is missing the required special-token IDs".into());
        }
        let mut word_to_id = HashMap::with_capacity(id_to_word.len());
        for (id, word) in id_to_word.iter().enumerate() {
            if word_to_id.insert(word.clone(), id).is_some() {
                return Err(format!("checkpoint tokenizer contains duplicate token '{word}'"));
            }
        }
        Ok(Self {
            word_to_id,
            id_to_word,
            configured_max_vocab,
            corpus_unique_words,
        })
    }

    pub(crate) fn checkpoint_vocabulary(&self) -> &[String] {
        &self.id_to_word
    }

    pub(crate) fn checkpoint_metadata(&self) -> (usize, usize) {
        (self.configured_max_vocab, self.corpus_unique_words)
    }

    /// Builds a frequency-ranked vocabulary from the supplied training corpus.
    /// Equal-frequency words are ordered alphabetically to keep IDs repeatable.
    pub fn build(corpus: &[&str], max_vocab: usize) -> Self {
        let mut frequencies: HashMap<String, usize> = HashMap::new();
        for sentence in corpus {
            for piece in lex(sentence) {
                if let Piece::Word(word) = piece {
                    *frequencies.entry(word).or_default() += 1;
                }
            }
        }

        let corpus_unique_words = frequencies.len();
        let learned_capacity = max_vocab.max(SPECIAL_TOKENS.len()) - SPECIAL_TOKENS.len();
        let mut ranked: Vec<(String, usize)> = frequencies.into_iter().collect();
        ranked.sort_by(|(word_a, count_a), (word_b, count_b)| {
            count_b.cmp(count_a).then_with(|| word_a.cmp(word_b))
        });

        let mut id_to_word: Vec<String> = SPECIAL_TOKENS.iter().map(|s| (*s).to_string()).collect();
        id_to_word.extend(
            ranked
                .into_iter()
                .take(learned_capacity)
                .map(|(word, _)| word),
        );
        let word_to_id = id_to_word
            .iter()
            .enumerate()
            .map(|(id, word)| (word.clone(), id))
            .collect();

        Self {
            word_to_id,
            id_to_word,
            configured_max_vocab: max_vocab,
            corpus_unique_words,
        }
    }

    /// Tokenizes words and wraps the sequence in BOS/EOS. Numeric values are
    /// stored in a parallel channel instead of splitting into digit tokens so
    /// the model can condition directly on exact computed evidence (for example,
    /// 20.43 remains one numeric value rather than five loosely related digits).
    pub fn tokenize(&self, text: &str) -> (Vec<usize>, Vec<Option<f64>>) {
        let mut ids = vec![self.bos_id()];
        let mut numbers = vec![None];
        for piece in lex(text) {
            match piece {
                Piece::Number(value) => {
                    ids.push(self.num_id());
                    numbers.push(Some(value));
                }
                Piece::Word(word) => {
                    ids.push(*self.word_to_id.get(&word).unwrap_or(&self.unk_id()));
                    numbers.push(None);
                }
            }
        }
        ids.push(self.eos_id());
        numbers.push(None);
        (ids, numbers)
    }

    /// Reconstructs readable words; numeric values use a placeholder because
    /// numeric reinsertion is a later component's responsibility.
    pub fn detokenize(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| match id {
                BOS | EOS | PAD => None,
                NUM => Some("[number]".to_string()),
                _ => Some(
                    self.id_to_word
                        .get(id)
                        .cloned()
                        .unwrap_or_else(|| "<UNK>".to_string()),
                ),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_word.len()
    }

    /// A compact build report, including kept and discarded corpus words.
    pub fn build_summary(&self) -> String {
        let learned_words = self.vocab_size() - SPECIAL_TOKENS.len();
        let omitted_words = self.corpus_unique_words.saturating_sub(learned_words);
        format!(
            "Vocabulary size: {} ({} special tokens, {} learned words, {} rare words mapped to <UNK>; configured maximum {})",
            self.vocab_size(),
            SPECIAL_TOKENS.len(),
            learned_words,
            omitted_words,
            self.configured_max_vocab
        )
    }

    pub const fn bos_id(&self) -> usize {
        BOS
    }
    pub const fn eos_id(&self) -> usize {
        EOS
    }
    pub const fn pad_id(&self) -> usize {
        PAD
    }
    pub const fn unk_id(&self) -> usize {
        UNK
    }
    pub const fn num_id(&self) -> usize {
        NUM
    }
}

#[derive(Debug, PartialEq)]
enum Piece {
    Word(String),
    Number(f64),
}

/// Splits punctuation-delimited words while recognizing the deliberately
/// narrow numeric grammar: optional minus, digits, and an optional decimal
/// point followed by digits. This keeps numeric punctuation such as `20.43`.
fn lex(text: &str) -> Vec<Piece> {
    let chars: Vec<char> = text.chars().collect();
    let mut pieces = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let starts_negative_number =
            chars[i] == '-' && chars.get(i + 1).is_some_and(|ch| ch.is_ascii_digit());
        if chars[i].is_ascii_digit() || starts_negative_number {
            let start = i;
            if starts_negative_number {
                i += 1;
            }
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
                i += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let number: String = chars[start..i].iter().collect();
            if let Ok(value) = number.parse::<f64>() {
                pieces.push(Piece::Number(value));
            }
        } else if chars[i].is_alphanumeric() {
            let start = i;
            while i < chars.len() && chars[i].is_alphanumeric() {
                i += 1;
            }
            pieces.push(Piece::Word(
                chars[start..i].iter().collect::<String>().to_lowercase(),
            ));
        } else {
            i += 1;
        }
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::Tokenizer;

    const CORPUS: [&str; 9] = [
        "The mean is 20.43",
        "The median is 18.00",
        "An increasing trend remains significant",
        "One outlier changes the slope",
        "The trend persists after excluding the outlier",
        "However the evidence is approximately stable",
        "The analysis confirms a rising pattern",
        "Mean and median summarize the values",
        "The trend is increasing at 20.43",
    ];

    fn tokenizer() -> Tokenizer {
        Tokenizer::build(&CORPUS, 2048)
    }

    #[test]
    fn corpus_round_trip_preserves_words_and_normalizes_case() {
        let tokenizer = tokenizer();
        let (ids, _) = tokenizer.tokenize("An increasing trend remains significant");
        assert_eq!(
            tokenizer.detokenize(&ids),
            "an increasing trend remains significant"
        );
    }

    #[test]
    fn numeric_tokens_have_parallel_float_values() {
        let tokenizer = tokenizer();
        let (ids, values) = tokenizer.tokenize("mean 20.43 and slope -3.7");
        assert_eq!(ids[2], tokenizer.num_id());
        assert_eq!(ids[5], tokenizer.num_id());
        assert_eq!(values[2], Some(20.43));
        assert_eq!(values[5], Some(-3.7));
        assert_eq!(values.len(), ids.len());
        assert!(values.iter().enumerate().all(|(i, value)| {
            if ids[i] == tokenizer.num_id() {
                value.is_some()
            } else {
                value.is_none()
            }
        }));
    }

    #[test]
    fn unknown_words_map_to_unk() {
        let tokenizer = tokenizer();
        let (ids, _) = tokenizer.tokenize("quasar");
        assert_eq!(ids[1], tokenizer.unk_id());
    }

    #[test]
    fn vocabulary_cap_keeps_frequent_words_and_drops_rare_words() {
        let corpus = [
            "common common common common common common common common",
            "rareword1 rareword2 rareword3 rareword4 rareword5 rareword6",
        ];
        let tokenizer = Tokenizer::build(&corpus, 10);
        assert!(tokenizer.vocab_size() <= 10);
        let (ids, _) = tokenizer.tokenize("common rareword6");
        assert_ne!(ids[1], tokenizer.unk_id());
        assert_eq!(ids[2], tokenizer.unk_id());
    }

    #[test]
    fn all_special_tokens_exist_even_for_an_empty_corpus() {
        let tokenizer = Tokenizer::build(&[], 5);
        assert_eq!(tokenizer.vocab_size(), 5);
        let ids = [
            tokenizer.bos_id(),
            tokenizer.eos_id(),
            tokenizer.pad_id(),
            tokenizer.unk_id(),
            tokenizer.num_id(),
        ];
        assert_eq!(ids, [0, 1, 2, 3, 4]);
        let (unknown, _) = tokenizer.tokenize("unseen");
        assert_eq!(unknown[1], tokenizer.unk_id());
        assert_eq!(tokenizer.detokenize(&ids), "<UNK> [number]");
    }

    #[test]
    fn inline_analytical_corpus_has_expected_vocabulary_size() {
        // The corpus contains 29 distinct words plus the five mandatory tokens.
        let tokenizer = tokenizer();
        assert_eq!(tokenizer.vocab_size(), 34);
    }
}
