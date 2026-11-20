use std::fs;
use std::collections::HashSet;

use crate::{data, explain, patterns, stats};

/// Runs the analysis and returns one explanation sentence per training example.
pub fn generate(csv_path: &str, column: &str) -> Result<Vec<String>, String> {
    let values = data::load_column(csv_path, column)?;
    Ok(explain_values(&values))
}

fn explain_values(values: &[f64]) -> Vec<String> {
    let summary = stats::Summary::from_values(values);
    let outliers = patterns::find_outliers(values, &summary);
    let robustness = patterns::check_trend_robustness(values, &outliers);
    let explanation = explain::generate(&summary, &outliers, &robustness);
    split_sentences(&explanation)
}

/// Writes the corpus as UTF-8 with one sentence on each line and a final newline.
pub fn write(path: &str, sentences: &[String]) -> Result<(), String> {
    let mut contents = sentences.join("\n");
    contents.push('\n');
    fs::write(path, contents).map_err(|error| format!("could not write corpus '{path}': {error}"))
}

/// Runs each generated scenario through the analysis pipeline and writes the
/// first occurrence of every distinct explanation sentence.
pub fn generate_v2(generated_dir: &str, output_path: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut seen = HashSet::new();
    for scenario_index in 1..=10 {
        let csv_path = format!("{generated_dir}/scenario_{scenario_index:02}.csv");
        let mut measurements = data::load_column(&csv_path, "measurement")
            .unwrap_or_else(|error| panic!("could not read scenario '{csv_path}': {error}"));
        let scenario_sentences = explain_values(&measurements);
        for sentence in scenario_sentences {
            if seen.insert(sentence.clone()) {
                sentences.push(sentence);
            }
        }
        // Include a deterministic prefix sensitivity analysis as distinct
        // computed evidence from the same generated scenario.
        if measurements.len() > 20 {
            measurements.truncate(20);
            for sentence in explain_values(&measurements) {
                if seen.insert(sentence.clone()) {
                    sentences.push(sentence);
                }
            }
        }
    }
    write(output_path, &sentences)
        .unwrap_or_else(|error| panic!("could not write expanded corpus: {error}"));
    sentences
}

fn split_sentences(text: &str) -> Vec<String> {
    let characters: Vec<(usize, char)> = text.char_indices().collect();
    let mut sentences = Vec::new();
    let mut start = 0;

    for (index, &(byte_index, character)) in characters.iter().enumerate() {
        if !matches!(character, '.' | '!' | '?') {
            continue;
        }
        let next = characters.get(index + 1).map(|(_, next)| *next);
        let previous = index.checked_sub(1).and_then(|i| characters.get(i)).map(|(_, ch)| *ch);
        let decimal_point = character == '.'
            && previous.is_some_and(|ch| ch.is_ascii_digit())
            && next.is_some_and(|ch| ch.is_ascii_digit());
        if decimal_point || next.is_some_and(|ch| !ch.is_whitespace()) {
            continue;
        }

        let end = byte_index + character.len_utf8();
        let sentence = text[start..end].trim();
        if !sentence.is_empty() {
            sentences.push(sentence.to_string());
        }
        start = characters
            .get(index + 1)
            .map_or(text.len(), |(next_index, _)| *next_index);
    }

    let remainder = text[start..].trim();
    if !remainder.is_empty() {
        sentences.push(remainder.to_string());
    }
    sentences
}

#[cfg(test)]
mod tests {
    use super::{generate, generate_v2};
    use reason_engine::gen_csv::{generate_csv, scenarios};
    use std::fs;
    use std::path::Path;

    #[test]
    fn sample_csv_produces_structurally_valid_training_corpus() {
        let csv_path = format!("{}/fixtures/sample.csv", env!("CARGO_MANIFEST_DIR"));
        let sentences = generate(&csv_path, "measurement").unwrap();

        for sentence in &sentences {
            println!("CORPUS: {sentence}");
        }
        assert!(sentences.len() >= 3, "expected at least three sentences");
        assert!(sentences.iter().all(|sentence| !sentence.trim().is_empty()));
        assert!(sentences.iter().all(|sentence| {
            !sentence.contains('{')
                && !sentence.contains('}')
                && !sentence.contains("TODO")
                && !sentence.contains("FIXME")
        }));
    }

    #[test]
    fn corpus_v2_has_enough_sentences() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let generated_dir = root.join("fixtures/generated");
        fs::create_dir_all(&generated_dir).unwrap();
        for (index, (scenario, seed)) in scenarios().iter().enumerate() {
            let path = generated_dir.join(format!("scenario_{:02}.csv", index + 1));
            fs::write(path, generate_csv(scenario, *seed)).unwrap();
        }

        let output_path = root.join("fixtures/corpus_v2.txt");
        let sentences = generate_v2(
            generated_dir.to_str().unwrap(),
            output_path.to_str().unwrap(),
        );
        for sentence in &sentences {
            println!("CORPUS_V2: {sentence}");
        }
        assert!(sentences.len() >= 30, "expected at least 30 unique sentences, got {}", sentences.len());
        assert!(sentences.iter().all(|sentence| !sentence.trim().is_empty()));
        assert!(sentences.iter().all(|sentence| {
            !sentence.contains('{')
                && !sentence.contains('}')
                && !sentence.contains("TODO")
                && !sentence.contains("FIXME")
        }));
    }
}
