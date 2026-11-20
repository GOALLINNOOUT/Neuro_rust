use reason_engine::tokenizer::Tokenizer;
use reason_engine::trainer::{diagnostic_last_epoch_max_gradient, set_diagnostics_enabled, Trainer, TrainingExample};
use reason_engine::transformer::Transformer;
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::time::Instant;
use reason_engine::attention::{diagnostic_max_score, reset_diagnostic_score};

const WORDS: &[&str] = &[
    "mean",
    "average",
    "readings",
    "describes",
    "median",
    "middle",
    "halfway",
    "midpoint",
    "variance",
    "measured",
    "sample",
    "standard",
    "deviation",
    "outlier",
    "outliers",
    "observations",
    "flagged",
    "values",
    "fences",
    "iqr",
    "fit",
    "increasing",
    "decreasing",
    "regression",
    "direction",
    "slope",
    "trend",
    "change",
];

const CONCEPT_WORDS: &[(&str, &[&str])] = &[
    ("Mean", &["mean", "average", "readings", "describes"]),
    ("Median", &["median", "middle", "halfway", "midpoint"]),
    (
        "Variance",
        &["variance", "measured", "sample", "standard", "deviation"],
    ),
    (
        "Outlier",
        &["outlier", "outliers", "observations", "flagged", "fences", "iqr"],
    ),
    (
        "Trend",
        &[
            "fit",
            "increasing",
            "decreasing",
            "regression",
            "direction",
            "slope",
            "trend",
            "change",
        ],
    ),
];

const SAME_CATEGORY_PAIRS: &[(&str, &str)] = &[
    ("mean", "average"),
    ("mean", "readings"),
    ("average", "describes"),
    ("median", "middle"),
    ("median", "halfway"),
    ("median", "midpoint"),
    ("middle", "halfway"),
    ("variance", "deviation"),
    ("variance", "standard"),
    ("variance", "sample"),
    ("measured", "variance"),
    ("outlier", "outliers"),
    ("outlier", "flagged"),
    ("outlier", "fences"),
    ("outlier", "iqr"),
    ("fit", "regression"),
    ("fit", "slope"),
    ("trend", "regression"),
    ("increasing", "decreasing"),
    ("direction", "change"),
];

const CROSS_CATEGORY_PAIRS: &[(&str, &str)] = &[
    ("mean", "slope"),
    ("mean", "outlier"),
    ("average", "regression"),
    ("median", "trend"),
    ("median", "increasing"),
    ("variance", "slope"),
    ("deviation", "outlier"),
    ("fences", "regression"),
    ("fit", "average"),
    ("direction", "median"),
];

fn main() -> Result<(), String> {
    let corpus_path = format!("{}/fixtures/corpus_v3.txt", env!("CARGO_MANIFEST_DIR"));
    let corpus_text = std::fs::read_to_string(&corpus_path)
        .map_err(|error| format!("could not read corpus '{corpus_path}': {error}"))?;
    let sentences: Vec<&str> = corpus_text.lines().collect();
    let tokenizer = Tokenizer::build(&sentences, 2048);
    let examples: Vec<_> = sentences
        .iter()
        .map(|sentence| TrainingExample::from_text(&tokenizer, sentence))
        .collect();
    let mut model = Transformer::new(tokenizer.vocab_size(), 128, 4, 512, 4, 64, 42)?;
    let average_loss = |model: &Transformer| -> f64 {
        examples
            .iter()
            .map(|example| {
                model
                    .loss(
                        &model
                            .forward(&example.token_ids, &example.numeric_values)
                            .unwrap(),
                        &example.token_ids,
                    )
            })
            .sum::<f64>()
            / examples.len() as f64
    };

    println!(
        "Corpus v3: {} sentences; {} tokenizer entries; architecture 128/4 heads/512/4 blocks; seed 42",
        examples.len(),
        tokenizer.vocab_size()
    );
    let started = Instant::now();
    for epoch in 1..=1000 {
        set_diagnostics_enabled(epoch >= 600);
        reset_diagnostic_score();
        if let Err(error) = Trainer::train(&mut model, &examples, 1, 0.1, 0) {
            if epoch >= 600 {
                let max_weight = model.diagnostic_extrema().into_iter().max_by(|a, b| a.1.total_cmp(&b.1));
                println!("DIAG epoch={epoch} FAILED max_weight={max_weight:?} max_abs_attention_score={} previous_epoch_max_abs_gradient={}", diagnostic_max_score(), diagnostic_last_epoch_max_gradient());
                let failed_index = error.split_whitespace().nth(2).unwrap_or("unknown").trim_end_matches(':');
                if let Ok(index) = failed_index.parse::<usize>() {
                    println!("DIAG failing_example={} text={}", index, sentences[index - 1]);
                }
            }
            println!("Training error at epoch {epoch}: {error}");
            io::stdout().flush().map_err(|flush_error| flush_error.to_string())?;
            return Err(format!("training failed at epoch {epoch}: {error}"));
        }
        if epoch >= 600 {
            let max_weight = model.diagnostic_extrema().into_iter().max_by(|a, b| a.1.total_cmp(&b.1));
            println!("DIAG epoch={epoch} max_weight={max_weight:?} max_abs_attention_score={} max_abs_gradient={}", diagnostic_max_score(), diagnostic_last_epoch_max_gradient());
        }
        let elapsed = started.elapsed().as_secs_f64();
        let remaining = elapsed / epoch as f64 * (1000 - epoch) as f64;
        println!(
            "Progress: epoch {epoch}/1000 ({:.1}%), elapsed {:.1}s, estimated remaining {:.1}s",
            epoch as f64 / 10.0,
            elapsed,
            remaining
        );
        io::stdout().flush().map_err(|error| error.to_string())?;
        if epoch == 100 || (epoch >= 200 && epoch % 200 == 0) {
            println!("Corpus v3 loss epoch {epoch}: {:.6}", average_loss(&model));
            io::stdout().flush().map_err(|error| error.to_string())?;
        }
    }

    let word_ids: Vec<usize> = WORDS
        .iter()
        .map(|word| {
            let (ids, _) = tokenizer.tokenize(word);
            if ids.len() != 3 || ids[1] == tokenizer.unk_id() {
                return Err(format!("analysis word '{word}' is not one in-vocabulary token"));
            }
            Ok(ids[1])
        })
        .collect::<Result<_, _>>()?;
    let (_, raw_embedding_rows, _, _) = model.forward_with_cache(
        &word_ids,
        &vec![None; word_ids.len()],
    )?;
    let vectors: HashMap<&str, Vec<f64>> = WORDS
        .iter()
        .copied()
        .zip(raw_embedding_rows)
        .collect();

    println!("\nSame-category template-associated word pairs:");
    let same_scores = print_pairs(SAME_CATEGORY_PAIRS, &vectors);
    println!("\nCross-category baseline word pairs:");
    let cross_scores = print_pairs(CROSS_CATEGORY_PAIRS, &vectors);
    let same_average = mean(&same_scores);
    let cross_average = mean(&cross_scores);
    println!("\nMean same-category pair cosine: {same_average:.9}");
    println!("Mean cross-category pair cosine: {cross_average:.9}");
    println!("Same minus cross: {:.9}", same_average - cross_average);

    let mut positions: HashMap<&str, HashSet<usize>> = HashMap::new();
    for sentence in &sentences {
        let (ids, _) = tokenizer.tokenize(sentence);
        for (body_position, token_id) in ids.iter().enumerate().skip(1).take(ids.len() - 2) {
            if let Some(word) = WORDS.iter().find(|word| tokenizer.tokenize(word).0[1] == *token_id) {
                positions.entry(word).or_default().insert(body_position - 1);
            }
        }
    }
    let mut same_position_pairs = Vec::new();
    for left_index in 0..CONCEPT_WORDS.len() {
        for right_index in left_index + 1..CONCEPT_WORDS.len() {
            for left_word in CONCEPT_WORDS[left_index].1 {
                for right_word in CONCEPT_WORDS[right_index].1 {
                    let shared_positions: Vec<usize> = positions
                        .get(left_word)
                        .into_iter()
                        .flatten()
                        .filter(|position| {
                            positions
                                .get(right_word)
                                .is_some_and(|set| set.contains(position))
                        })
                        .copied()
                        .collect();
                    if !shared_positions.is_empty() {
                        same_position_pairs.push((
                            *left_word,
                            *right_word,
                            shared_positions,
                            cosine(&vectors[left_word], &vectors[right_word]),
                        ));
                    }
                }
            }
        }
    }
    same_position_pairs.sort_by(|left, right| {
        right
            .3
            .total_cmp(&left.3)
            .then_with(|| left.0.cmp(right.0))
            .then_with(|| left.1.cmp(right.1))
    });
    println!("\nCross-category pairs observed at the same sentence-token position:");
    for (left, right, shared, similarity) in &same_position_pairs {
        println!("{left} vs {right} | positions={shared:?} | cosine={similarity:.9}");
    }
    let position_scores: Vec<f64> = same_position_pairs.iter().map(|pair| pair.3).collect();
    println!("Same-position cross-category pair count: {}", position_scores.len());
    println!("Same-position cross-category mean cosine: {:.9}", mean(&position_scores));
    println!("Final corpus v3 average loss: {:.6}", average_loss(&model));
    Ok(())
}

fn print_pairs(pairs: &[(&str, &str)], vectors: &HashMap<&str, Vec<f64>>) -> Vec<f64> {
    let mut scores = Vec::with_capacity(pairs.len());
    for &(left, right) in pairs {
        let similarity = cosine(&vectors[left], &vectors[right]);
        println!("{left} vs {right}: {similarity:.9}");
        scores.push(similarity);
    }
    scores
}

fn cosine(left: &[f64], right: &[f64]) -> f64 {
    let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();
    let left_norm = left.iter().map(|value| value * value).sum::<f64>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f64>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm * right_norm)
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}
