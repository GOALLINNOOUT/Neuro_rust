use crate::patterns::{
    Direction, MIN_TREND_SAMPLE_SIZE, Outlier, RobustnessClass, TrendRobustness,
};
use crate::stats::Summary;

pub fn generate(summary: &Summary, outliers: &[Outlier], robustness: &TrendRobustness) -> String {
    let mut sentences = vec![format!(
        "The mean is {:.2} and the median is {:.2} (sample variance {:.2}, standard deviation {:.2}).",
        summary.mean, summary.median, summary.variance, summary.standard_deviation
    )];
    if outliers.is_empty() {
        sentences.push("No outliers found by the 1.5×IQR rule.".to_string());
    } else {
        let details = outliers
            .iter()
            .map(|o| format!("{} at position {}", format_number(o.value), o.index))
            .collect::<Vec<_>>()
            .join(", ");
        let (subject, location_label) = if outliers.len() == 1 {
            ("this outlier", "zero-based position")
        } else {
            ("these outliers", "zero-based positions")
        };
        sentences.push(format!(
            "The 1.5×IQR rule identified {subject} ({location_label}): {details}."
        ));
    }
    if robustness.class == RobustnessClass::InsufficientData {
        let remaining = robustness.without_outliers.sample_size;
        if !outliers.is_empty() && remaining < MIN_TREND_SAMPLE_SIZE {
            sentences.push(format!(
                "There isn't enough data after excluding the outliers to assess a trend reliably; at least {MIN_TREND_SAMPLE_SIZE} observations are required."
            ));
        } else {
            sentences.push(format!(
                "There isn't enough data to assess a trend reliably; at least {MIN_TREND_SAMPLE_SIZE} observations are required."
            ));
        }
        return sentences.join(" ");
    }

    sentences.push(format!("{}.", describe_fit(&robustness.full)));
    if !outliers.is_empty() {
        match robustness.class {
            RobustnessClass::Confirmed => sentences.push(format!(
                "The trend remains significant after excluding the outliers. {}.",
                describe_fit(&robustness.without_outliers)
            )),
            RobustnessClass::Weakened => sentences.push(format!(
                "Significance appears only after excluding the outliers. {}.",
                describe_fit(&robustness.without_outliers)
            )),
            RobustnessClass::Artifact => sentences.push(
                "The full-data trend appears to depend on the outlier(s) or changes direction after their removal, so it may not reflect the broader pattern.".to_string(),
            ),
            RobustnessClass::NoReliableTrend => sentences.push(
                "Neither the full data nor the data excluding outliers shows a reliable trend under the approximate |t| > 2 rule.".to_string(),
            ),
            RobustnessClass::InsufficientData | RobustnessClass::Unchecked => {}
        }
    } else if robustness.class == RobustnessClass::NoReliableTrend {
        sentences
            .push("No reliable trend is detected under the approximate |t| > 2 rule.".to_string());
    }
    sentences.join(" ")
}

fn describe_fit(trend: &crate::patterns::Trend) -> String {
    let direction = match trend.direction {
        Direction::Increasing => "increasing",
        Direction::Decreasing => "decreasing",
        Direction::Flat => "flat",
    };
    match (trend.standard_error, trend.t_statistic) {
        (Some(se), Some(t)) => format!(
            "The fit is {} (slope {:.3} per observation, SE {:.3}, t={:.2}; significant by the approximate |t| > 2 rule)",
            direction, trend.slope, se, t
        ),
        (Some(se), _) => format!(
            "The fit is {} (slope {:.3} per observation, SE {:.3}; not significant by the approximate |t| > 2 rule)",
            direction, trend.slope, se
        ),
        (None, _) => format!(
            "The fit is {} (slope {:.3} per observation; standard error unavailable with {} observations)",
            direction, trend.slope, trend.sample_size
        ),
    }
}

fn format_number(value: f64) -> String {
    let mut s = format!("{value:.2}");
    while s.contains('.') && s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    s
}
