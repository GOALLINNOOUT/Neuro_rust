//! Phase 3, Stage 1 rules-only question dispatcher.
//!
//! Design note: this module performs lexical rule matching only. It does not
//! call the Transformer, consult memory, inspect a dataset, or execute a
//! computation. The caller supplies the available column names. A column is
//! returned only when exactly one supplied name appears as a whole-word or
//! contiguous multiword match in the question; no match and multiple matches
//! both yield `None`.
//!
//! | Intent | Example lexical cues | Destination |
//! | --- | --- | --- |
//! | Mean | `mean`, `average`, `arithmetic mean` | `stats::mean` |
//! | Median | `median`, `typical`, `middle value`, `central value` | `stats::median` |
//! | Variance | `variance`, `variability`, `dispersion`, `spread`, `vary` | `stats::variance` |
//! | Outlier check | `outlier`, `anomaly`, `unusual`, `extreme`, `aberrant` | `patterns::find_outliers` (IQR) |
//! | Trend check | `trend`, `trending`, `slope`, `over time`, `upward`, `downward`, `increase`, `decrease`, `drift` | `patterns::detect_trend` (OLS) |
//!
//! A question with cues for more than one computation is treated as
//! unrecognized instead of choosing an arbitrary rule. Cues are matched after
//! lowercasing and splitting into alphanumeric words. This is the Rules layer
//! only; Stage 2 may later connect results to Memory/Neural/Transformer.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchResult {
    Mean {
        target_column: Option<String>,
    },
    Median {
        target_column: Option<String>,
    },
    /// Included because Stage 1 explicitly requires variance dispatch.
    Variance {
        target_column: Option<String>,
    },
    OutlierCheck {
        target_column: Option<String>,
    },
    TrendCheck {
        target_column: Option<String>,
    },
    Unrecognized {
        target_column: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Intent {
    Mean,
    Median,
    Variance,
    Outlier,
    Trend,
}

const MEAN_CUES: &[&str] = &["mean", "average"];
const MEDIAN_CUES: &[&str] = &["median", "typical", "middle value", "central value"];
const VARIANCE_CUES: &[&str] = &[
    "variance",
    "variability",
    "dispersion",
    "spread",
    "vary",
    "varies",
];
const OUTLIER_CUES: &[&str] = &[
    "outlier",
    "outliers",
    "anomaly",
    "anomalies",
    "unusual",
    "extreme",
    "aberrant",
];
const TREND_CUES: &[&str] = &[
    "trend",
    "trending",
    "slope",
    "over time",
    "upward",
    "downward",
    "increase",
    "increases",
    "increasing",
    "decrease",
    "decreases",
    "decreasing",
    "drift",
    "pattern over time",
];

/// Classifies a question using fixed phrase rules and resolves a named column
/// against `available_columns`. No model or external crate is used.
pub fn dispatch(question: &str, available_columns: &[String]) -> DispatchResult {
    let question_words = words(question);
    let target_column = find_target_column(&question_words, available_columns);
    let intents = [
        (Intent::Mean, MEAN_CUES),
        (Intent::Median, MEDIAN_CUES),
        (Intent::Variance, VARIANCE_CUES),
        (Intent::Outlier, OUTLIER_CUES),
        (Intent::Trend, TREND_CUES),
    ];
    let matched: Vec<Intent> = intents
        .iter()
        .filter_map(|(intent, cues)| {
            cues.iter()
                .any(|cue| contains_phrase(&question_words, &words(cue)))
                .then_some(*intent)
        })
        .collect();

    if matched.len() != 1 {
        return DispatchResult::Unrecognized { target_column };
    }

    match matched[0] {
        Intent::Mean => DispatchResult::Mean { target_column },
        Intent::Median => DispatchResult::Median { target_column },
        Intent::Variance => DispatchResult::Variance { target_column },
        Intent::Outlier => DispatchResult::OutlierCheck { target_column },
        Intent::Trend => DispatchResult::TrendCheck { target_column },
    }
}

fn find_target_column(question_words: &[String], available_columns: &[String]) -> Option<String> {
    let matches: Vec<&String> = available_columns
        .iter()
        .filter(|column| {
            let column_words = words(column);
            !column_words.is_empty() && contains_phrase(question_words, &column_words)
        })
        .collect();
    (matches.len() == 1).then(|| matches[0].clone())
}

fn contains_phrase(haystack: &[String], needle: &[String]) -> bool {
    !needle.is_empty()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

fn words(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if character.is_alphanumeric() {
            current.extend(character.to_lowercase());
        } else if !current.is_empty() {
            result.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{dispatch, DispatchResult};

    fn result(kind: fn(Option<String>) -> DispatchResult, column: Option<&str>) -> DispatchResult {
        kind(column.map(str::to_string))
    }

    fn mean(column: Option<String>) -> DispatchResult {
        DispatchResult::Mean {
            target_column: column,
        }
    }
    fn median(column: Option<String>) -> DispatchResult {
        DispatchResult::Median {
            target_column: column,
        }
    }
    fn variance(column: Option<String>) -> DispatchResult {
        DispatchResult::Variance {
            target_column: column,
        }
    }
    fn outlier(column: Option<String>) -> DispatchResult {
        DispatchResult::OutlierCheck {
            target_column: column,
        }
    }
    fn trend(column: Option<String>) -> DispatchResult {
        DispatchResult::TrendCheck {
            target_column: column,
        }
    }
    fn unknown(column: Option<String>) -> DispatchResult {
        DispatchResult::Unrecognized {
            target_column: column,
        }
    }

    #[test]
    fn held_out_plain_english_dispatch_cases() {
        let columns = vec!["measurement".to_string(), "index".to_string()];
        let cases: [(&str, fn(Option<String>) -> DispatchResult, Option<&str>); 20] = [
            (
                "Could you calculate the average for measurement?",
                mean,
                Some("measurement"),
            ),
            (
                "Give me the arithmetic mean of measurement.",
                mean,
                Some("measurement"),
            ),
            (
                "What value sits in the middle for measurement?",
                median,
                Some("measurement"),
            ),
            (
                "Which reading is typical for measurement?",
                median,
                Some("measurement"),
            ),
            (
                "Please report measurement's median.",
                median,
                Some("measurement"),
            ),
            (
                "How much variability is present in measurement?",
                variance,
                Some("measurement"),
            ),
            (
                "Calculate the variance for measurement.",
                variance,
                Some("measurement"),
            ),
            (
                "Describe the dispersion of measurement.",
                variance,
                Some("measurement"),
            ),
            (
                "Should I flag extreme readings in measurement?",
                outlier,
                Some("measurement"),
            ),
            (
                "Does measurement contain any anomalies?",
                outlier,
                Some("measurement"),
            ),
            (
                "Check measurement for outliers.",
                outlier,
                Some("measurement"),
            ),
            (
                "Does measurement rise across the sample?",
                trend,
                Some("measurement"),
            ),
            (
                "Estimate a slope for measurement.",
                trend,
                Some("measurement"),
            ),
            (
                "Is measurement drifting downward?",
                trend,
                Some("measurement"),
            ),
            (
                "Does measurement change over time?",
                trend,
                Some("measurement"),
            ),
            ("Tell me about measurement.", unknown, Some("measurement")),
            ("Is measurement interesting?", unknown, Some("measurement")),
            (
                "Compare the average and the trend for measurement.",
                unknown,
                Some("measurement"),
            ),
            (
                "What is typical and unusual about measurement?",
                unknown,
                Some("measurement"),
            ),
            ("Is there a clear pattern?", unknown, None),
        ];

        let mut correct = 0;
        for (question, expected_kind, expected_column) in cases {
            let expected = result(expected_kind, expected_column);
            let actual = dispatch(question, &columns);
            let is_correct = actual == expected;
            println!("{question:?} => {actual:?} | correct={is_correct}");
            correct += usize::from(is_correct);
        }
        println!("Held-out dispatch accuracy: {correct}/{}", cases.len());
        assert!(
            correct * 100 >= cases.len() * 90,
            "held-out accuracy below 90%: {correct}/{}",
            cases.len()
        );
    }

    #[test]
    fn ambiguous_column_mentions_leave_target_unset() {
        let columns = vec!["measurement".to_string(), "index".to_string()];
        assert_eq!(
            dispatch("Compare the average of measurement and index", &columns),
            DispatchResult::Mean {
                target_column: None
            }
        );
    }

    #[test]
    fn unknown_column_words_do_not_become_targets() {
        let columns = vec!["measurement".to_string(), "index".to_string()];
        assert_eq!(
            dispatch("Find the average of temperature", &columns),
            DispatchResult::Mean {
                target_column: None
            }
        );
    }
}
