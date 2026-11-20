//! Phase 3, Stage 1b lexical memory layer.
//!
//! Similarity uses cosine similarity over bag-of-words term counts, omitting a
//! small fixed set of grammatical stopwords so shared words like "the" do not
//! swamp intent-bearing overlap. This needs only the standard library,
//! accounts for repeated terms, and avoids introducing trained embeddings
//! before the Neural/Transformer stage exists. It deliberately remains a
//! lexical baseline: paraphrases with no shared terms may not be recovered.
//! The fixed examples are written as natural requests rather than copying the
//! dispatcher's cue list.

use crate::dispatcher::{self, DispatchResult};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Intent {
    Mean,
    Median,
    Variance,
    Outlier,
    Trend,
}

/// Returns the k highest-scoring memory entries, in descending similarity
/// order. Equal scores retain the memory bank's insertion order.
pub fn recall(question: &str, k: usize) -> Vec<(Intent, f64)> {
    let query = term_counts(question);
    let mut scored: Vec<(usize, Intent, f64)> = memory_bank()
        .into_iter()
        .enumerate()
        .map(|(index, (example, intent))| {
            (
                index,
                intent,
                cosine_similarity(&query, &term_counts(&example)),
            )
        })
        .collect();
    scored.sort_by(|left, right| {
        right
            .2
            .total_cmp(&left.2)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored
        .into_iter()
        .take(k)
        .map(|(_, intent, similarity)| (intent, similarity))
        .collect()
}

/// Votes over the k nearest neighbors that meet `min_similarity`.
/// Returns None for an empty qualifying set or a tie between intents.
pub fn memory_vote(question: &str, k: usize, min_similarity: f64) -> Option<Intent> {
    let mut counts: HashMap<Intent, usize> = HashMap::new();
    for (intent, similarity) in recall(question, k) {
        if similarity >= min_similarity {
            *counts.entry(intent).or_default() += 1;
        }
    }
    let max_votes = counts.values().copied().max()?;
    let winners: Vec<Intent> = counts
        .into_iter()
        .filter_map(|(intent, votes)| (votes == max_votes).then_some(intent))
        .collect();
    (winners.len() == 1).then_some(winners[0])
}

/// Applies Rules first. Memory runs only for rule-layer `Unrecognized`
/// results, and reuses the target column already resolved by the dispatcher.
pub fn dispatch_with_memory(
    question: &str,
    available_columns: &[String],
    k: usize,
    min_similarity: f64,
) -> DispatchResult {
    let rules_result = dispatcher::dispatch(question, available_columns);
    let target_column = match rules_result {
        DispatchResult::Unrecognized { target_column } => target_column,
        recognized => return recognized,
    };

    match memory_vote(question, k, min_similarity) {
        Some(Intent::Mean) => DispatchResult::Mean { target_column },
        Some(Intent::Median) => DispatchResult::Median { target_column },
        Some(Intent::Variance) => DispatchResult::Variance { target_column },
        Some(Intent::Outlier) => DispatchResult::OutlierCheck { target_column },
        Some(Intent::Trend) => DispatchResult::TrendCheck { target_column },
        None => DispatchResult::Unrecognized { target_column },
    }
}

fn memory_bank() -> Vec<(String, Intent)> {
    [
        (
            "Pool all observations into one representative figure",
            Intent::Mean,
        ),
        (
            "Add every reading then share the total evenly across records",
            Intent::Mean,
        ),
        (
            "Summarize the entire collection with a single balanced amount",
            Intent::Mean,
        ),
        (
            "Sort the observations and choose the middle reading",
            Intent::Median,
        ),
        (
            "Arrange all values and select the halfway item",
            Intent::Median,
        ),
        (
            "Identify the point that leaves equal counts on either side",
            Intent::Median,
        ),
        (
            "Measure how far readings wander from their common center",
            Intent::Variance,
        ),
        (
            "Describe squared departures among the observations",
            Intent::Variance,
        ),
        (
            "Quantify the width of variation around the group center",
            Intent::Variance,
        ),
        (
            "Find records isolated from the main cluster",
            Intent::Outlier,
        ),
        (
            "Mark observations far outside the usual bounds",
            Intent::Outlier,
        ),
        (
            "Locate values that stand apart from neighboring points",
            Intent::Outlier,
        ),
        (
            "Check whether values rise steadily as the row index advances",
            Intent::Trend,
        ),
        (
            "See whether readings climb or fall along sequence order",
            Intent::Trend,
        ),
        (
            "Compare early and late readings across ordered rows to establish direction",
            Intent::Trend,
        ),
    ]
    .into_iter()
    .map(|(question, intent)| (question.to_string(), intent))
    .collect()
}

fn term_counts(text: &str) -> HashMap<String, f64> {
    const STOPWORDS: &[&str] = &[
        "a", "about", "after", "all", "along", "an", "and", "any", "as", "across", "be", "can",
        "could", "does", "do", "for", "from", "how", "i", "in", "into", "is", "it", "me", "of",
        "on", "or", "please", "should", "that", "the", "their", "then", "there", "these", "this",
        "to", "up", "what", "which", "with", "would", "you",
    ];
    let mut counts = HashMap::new();
    let mut term = String::new();
    for character in text.chars() {
        if character.is_alphanumeric() {
            term.extend(character.to_lowercase());
        } else if !term.is_empty() {
            let completed = std::mem::take(&mut term);
            if !STOPWORDS.contains(&completed.as_str()) {
                *counts.entry(completed).or_insert(0.0) += 1.0;
            }
        }
    }
    if !term.is_empty() {
        if !STOPWORDS.contains(&term.as_str()) {
            *counts.entry(term).or_insert(0.0) += 1.0;
        }
    }
    counts
}

fn cosine_similarity(left: &HashMap<String, f64>, right: &HashMap<String, f64>) -> f64 {
    let left_norm = left.values().map(|value| value * value).sum::<f64>().sqrt();
    let right_norm = right
        .values()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    let dot = left
        .iter()
        .filter_map(|(term, left_count)| {
            right.get(term).map(|right_count| left_count * right_count)
        })
        .sum::<f64>();
    dot / (left_norm * right_norm)
}

#[cfg(test)]
mod tests {
    use super::{dispatch_with_memory, memory_vote, Intent};
    use crate::dispatcher::DispatchResult;

    fn expected(intent: Intent, column: Option<&str>) -> DispatchResult {
        let target_column = column.map(str::to_string);
        match intent {
            Intent::Mean => DispatchResult::Mean { target_column },
            Intent::Median => DispatchResult::Median { target_column },
            Intent::Variance => DispatchResult::Variance { target_column },
            Intent::Outlier => DispatchResult::OutlierCheck { target_column },
            Intent::Trend => DispatchResult::TrendCheck { target_column },
        }
    }

    fn unrecognized(column: Option<&str>) -> DispatchResult {
        DispatchResult::Unrecognized {
            target_column: column.map(str::to_string),
        }
    }

    fn run_set(name: &str, cases: &[(&str, DispatchResult)], k: usize, threshold: f64) -> usize {
        let columns = vec!["measurement".to_string(), "index".to_string()];
        let mut correct = 0;
        println!("{name}");
        for (question, expected) in cases {
            let actual = dispatch_with_memory(question, &columns, k, threshold);
            let is_correct = &actual == expected;
            println!("{question:?} => {actual:?} | correct={is_correct}");
            correct += usize::from(is_correct);
        }
        println!("{name} accuracy: {correct}/{}", cases.len());
        correct
    }

    #[test]
    fn original_dispatcher_cases_through_rules_then_memory() {
        use Intent::{Mean as M, Median as Md, Outlier as O, Trend as T, Variance as V};
        let cases = [
            (
                "Could you calculate the average for measurement?",
                expected(M, Some("measurement")),
            ),
            (
                "Give me the arithmetic mean of measurement.",
                expected(M, Some("measurement")),
            ),
            (
                "What value sits in the middle for measurement?",
                expected(Md, Some("measurement")),
            ),
            (
                "Which reading is typical for measurement?",
                expected(Md, Some("measurement")),
            ),
            (
                "Please report measurement's median.",
                expected(Md, Some("measurement")),
            ),
            (
                "How much variability is present in measurement?",
                expected(V, Some("measurement")),
            ),
            (
                "Calculate the variance for measurement.",
                expected(V, Some("measurement")),
            ),
            (
                "Describe the dispersion of measurement.",
                expected(V, Some("measurement")),
            ),
            (
                "Should I flag extreme readings in measurement?",
                expected(O, Some("measurement")),
            ),
            (
                "Does measurement contain any anomalies?",
                expected(O, Some("measurement")),
            ),
            (
                "Check measurement for outliers.",
                expected(O, Some("measurement")),
            ),
            (
                "Does measurement rise across the sample?",
                expected(T, Some("measurement")),
            ),
            (
                "Estimate a slope for measurement.",
                expected(T, Some("measurement")),
            ),
            (
                "Is measurement drifting downward?",
                expected(T, Some("measurement")),
            ),
            (
                "Does measurement change over time?",
                expected(T, Some("measurement")),
            ),
            (
                "Tell me about measurement.",
                unrecognized(Some("measurement")),
            ),
            (
                "Is measurement interesting?",
                unrecognized(Some("measurement")),
            ),
            (
                "Compare the average and the trend for measurement.",
                unrecognized(Some("measurement")),
            ),
            (
                "What is typical and unusual about measurement?",
                unrecognized(Some("measurement")),
            ),
            ("Is there a clear pattern?", unrecognized(None)),
        ];
        let correct = run_set("Set A: original 20", &cases, 1, 0.18);
        assert_eq!(correct, 20);
    }

    #[test]
    fn fresh_memory_paraphrases_and_unrecognizable_question() {
        use Intent::{Mean as M, Median as Md, Outlier as O, Trend as T, Variance as V};
        let cases = [
            (
                "Turn all readings into one pooled figure that speaks for the collection.",
                expected(M, None),
            ),
            (
                "What single summary amount stands for the complete set of entries?",
                expected(M, None),
            ),
            (
                "After ordering the records, pick the item at the halfway mark.",
                expected(Md, None),
            ),
            (
                "Choose the reading that leaves matching counts on both sides after sorting.",
                expected(Md, None),
            ),
            (
                "How far do measurements wander away from their shared center?",
                expected(V, None),
            ),
            (
                "Measure squared departures among the records.",
                expected(V, None),
            ),
            (
                "Locate records that sit apart from the main cluster.",
                expected(O, None),
            ),
            (
                "Mark readings far beyond the ordinary bounds.",
                expected(O, None),
            ),
            (
                "Do the values rise along the ordered rows?",
                expected(T, None),
            ),
            (
                "Compare early and late values across the ordered rows.",
                expected(T, None),
            ),
            ("Can you unlock the blue folder?", unrecognized(None)),
        ];
        let correct = run_set("Set B: fresh paraphrases", &cases, 1, 0.18);
        assert_eq!(correct, cases.len());
    }

    #[test]
    fn low_similarity_question_does_not_receive_a_memory_vote() {
        assert_eq!(memory_vote("unlock blue folder", 1, 0.18), None);
    }
}
