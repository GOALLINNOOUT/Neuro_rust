use reason_engine::dispatcher::DispatchResult;
use reason_engine::memory::{dispatch_with_memory, recall, Intent};
use std::collections::{HashMap, HashSet};

// Mirrors memory.rs::term_counts exactly so the integration test can report
// its extracted keys without changing that private function or memory.rs.
fn actual_term_counts(text: &str) -> HashMap<String, f64> {
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
    if !term.is_empty() && !STOPWORDS.contains(&term.as_str()) {
        *counts.entry(term).or_insert(0.0) += 1.0;
    }
    counts
}

fn add_terms(destination: &mut HashSet<String>, text: &str) {
    destination.extend(actual_term_counts(text).into_keys());
}

#[derive(Clone, Copy, Debug)]
struct Case {
    question: &'static str,
    expected: Intent,
}

const RULE_CUES: &[&str] = &[
    "mean",
    "average",
    "median",
    "typical",
    "middle value",
    "central value",
    "variance",
    "variability",
    "dispersion",
    "spread",
    "vary",
    "varies",
    "outlier",
    "outliers",
    "anomaly",
    "anomalies",
    "unusual",
    "extreme",
    "aberrant",
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

const SET_A: &[&str] = &[
    "Could you calculate the average for measurement?",
    "Give me the arithmetic mean of measurement.",
    "What value sits in the middle for measurement?",
    "Which reading is typical for measurement?",
    "Please report measurement's median.",
    "How much variability is present in measurement?",
    "Calculate the variance for measurement.",
    "Describe the dispersion of measurement.",
    "Should I flag extreme readings in measurement?",
    "Does measurement contain any anomalies?",
    "Check measurement for outliers.",
    "Does measurement rise across the sample?",
    "Estimate a slope for measurement.",
    "Is measurement drifting downward?",
    "Does measurement change over time?",
    "Tell me about measurement.",
    "Is measurement interesting?",
    "Compare the average and the trend for measurement.",
    "What is typical and unusual about measurement?",
    "Is there a clear pattern?",
];

const SET_B: &[&str] = &[
    "Turn all readings into one pooled figure that speaks for the collection.",
    "What single summary amount stands for the complete set of entries?",
    "After ordering the records, pick the item at the halfway mark.",
    "Choose the reading that leaves matching counts on both sides after sorting.",
    "How far do measurements wander away from their shared center?",
    "Measure squared departures among the records.",
    "Locate records that sit apart from the main cluster.",
    "Mark readings far beyond the ordinary bounds.",
    "Do the values rise along the ordered rows?",
    "Compare early and late values across the ordered rows.",
    "Can you unlock the blue folder?",
];

const CASES: [Case; 10] = [
    Case {
        question: "Derive the population expectation from this vector.",
        expected: Intent::Mean,
    },
    Case {
        question: "State the barycenter of an empirical distribution.",
        expected: Intent::Mean,
    },
    Case {
        question: "Retrieve the 50th quantile for the series.",
        expected: Intent::Median,
    },
    Case {
        question: "Return the second quartile of the dataset.",
        expected: Intent::Median,
    },
    Case {
        question: "State a population second moment relative to expectation.",
        expected: Intent::Variance,
    },
    Case {
        question: "Gauge heterogeneity with a secondary statistical moment.",
        expected: Intent::Variance,
    },
    Case {
        question: "Does any datum deviate from its cohort?",
        expected: Intent::Outlier,
    },
    Case {
        question: "Might a specimen be rogue relative to peers?",
        expected: Intent::Outlier,
    },
    Case {
        question: "Does a trajectory exhibit persistent escalation across chronological phases?",
        expected: Intent::Trend,
    },
    Case {
        question: "Could the response show systematic recession through successive cycles?",
        expected: Intent::Trend,
    },
];

#[test]
fn adversarial_zero_overlap_paraphrases_at_current_settings() {
    let mut source_terms = HashSet::new();
    for phrase in RULE_CUES {
        add_terms(&mut source_terms, phrase);
    }
    for phrase in SET_A.iter().chain(SET_B) {
        add_terms(&mut source_terms, phrase);
    }
    // These are the bank's 15 canonical examples, copied here only for the
    // independent cross-source overlap check.
    for phrase in [
        "Pool all observations into one representative figure",
        "Add every reading then share the total evenly across records",
        "Summarize the entire collection with a single balanced amount",
        "Sort the observations and choose the middle reading",
        "Arrange all values and select the halfway item",
        "Identify the point that leaves equal counts on either side",
        "Measure how far readings wander from their common center",
        "Describe squared departures among the observations",
        "Quantify the width of variation around the group center",
        "Find records isolated from the main cluster",
        "Mark observations far outside the usual bounds",
        "Locate values that stand apart from neighboring points",
        "Check whether values rise steadily as the row index advances",
        "See whether readings climb or fall along sequence order",
        "Compare early and late readings across ordered rows to establish direction",
    ] {
        add_terms(&mut source_terms, phrase);
    }

    let mut zero_similarity = 0;
    let mut partial_overlap = 0;
    let columns = vec!["measurement".to_string(), "index".to_string()];

    for case in CASES {
        let counts = actual_term_counts(case.question);
        let mut keys: Vec<String> = counts.keys().cloned().collect();
        keys.sort();
        let mut overlaps: Vec<String> = keys
            .iter()
            .filter(|term| source_terms.contains(*term))
            .cloned()
            .collect();
        overlaps.sort();

        let top_one = recall(case.question, 1);
        let actual = dispatch_with_memory(case.question, &columns, 1, 0.18);
        if top_one
            .first()
            .is_some_and(|(_, similarity)| *similarity == 0.0)
        {
            zero_similarity += 1;
        } else {
            partial_overlap += 1;
        }

        println!("Question: {}", case.question);
        println!("term_counts keys: {keys:?}");
        println!("cross-source overlaps: {overlaps:?}");
        println!("top-1 recall(): {top_one:?}");
        println!("dispatch_with_memory(): {actual:?}");
        println!(
            "expected intent: {:?}; actual result: {actual:?}",
            case.expected
        );

        assert!(overlaps.is_empty(), "unexpected term overlap: {overlaps:?}");
        assert_eq!(top_one, [(Intent::Mean, 0.0)]);
        assert_eq!(
            actual,
            DispatchResult::Unrecognized {
                target_column: None
            }
        );
    }

    println!("Effectively zero similarity to every bank entry: {zero_similarity}/10");
    println!("Partial/accidental bank overlap: {partial_overlap}/10");
    assert_eq!(zero_similarity, 10);
    assert_eq!(partial_overlap, 0);
}
