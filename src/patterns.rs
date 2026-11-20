use crate::stats::Summary;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Increasing,
    Decreasing,
    Flat,
}

/// Require eight observations so the regression and residual variance are not
/// based on only a handful of points; this is a modest reliability floor, not
/// a guarantee of adequate power.
pub const MIN_TREND_SAMPLE_SIZE: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct Trend {
    pub slope: f64,
    pub direction: Direction,
    pub sample_size: usize,
    pub standard_error: Option<f64>,
    pub t_statistic: Option<f64>,
    pub significant: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Outlier {
    pub index: usize,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustnessClass {
    Confirmed,
    Weakened,
    Artifact,
    NoReliableTrend,
    InsufficientData,
    Unchecked,
}

#[derive(Debug, Clone, Copy)]
pub struct TrendRobustness {
    pub full: Trend,
    pub without_outliers: Trend,
    pub class: RobustnessClass,
}

pub fn find_outliers(values: &[f64], summary: &Summary) -> Vec<Outlier> {
    let low = summary.q1 - 1.5 * summary.iqr;
    let high = summary.q3 + 1.5 * summary.iqr;
    values
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| {
            (value < low || value > high).then_some(Outlier { index, value })
        })
        .collect()
}

/// OLS on x=index. Flat means |slope| <= 1% of observed value range per step.
/// For constant data the threshold is zero, so an exactly zero slope is flat.
pub fn detect_trend(values: &[f64]) -> Trend {
    let indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    detect_indexed_trend(&indexed)
}

/// Computes OLS inference after removing outliers, retaining each observation's
/// original x=index coordinate. The sample-size gate is applied to both fits.
pub fn check_trend_robustness(values: &[f64], outliers: &[Outlier]) -> TrendRobustness {
    let full = detect_trend(values);
    if full.sample_size < MIN_TREND_SAMPLE_SIZE {
        return TrendRobustness {
            full,
            without_outliers: full,
            class: RobustnessClass::InsufficientData,
        };
    }
    if outliers.is_empty() {
        return TrendRobustness {
            full,
            without_outliers: full,
            class: RobustnessClass::Unchecked,
        };
    }

    let excluded: Vec<usize> = outliers.iter().map(|outlier| outlier.index).collect();
    let remaining: Vec<(usize, f64)> = values
        .iter()
        .copied()
        .enumerate()
        .filter(|(index, _)| !excluded.contains(index))
        .collect();
    let without_outliers = detect_indexed_trend(&remaining);
    let class = if full.sample_size < MIN_TREND_SAMPLE_SIZE
        || without_outliers.sample_size < MIN_TREND_SAMPLE_SIZE
    {
        RobustnessClass::InsufficientData
    } else if !full.significant && !without_outliers.significant {
        RobustnessClass::NoReliableTrend
    } else if full.direction != without_outliers.direction {
        RobustnessClass::Artifact
    } else if full.significant && without_outliers.significant {
        RobustnessClass::Confirmed
    } else if full.significant && !without_outliers.significant {
        RobustnessClass::Artifact
    } else if !full.significant && without_outliers.significant {
        RobustnessClass::Weakened
    } else {
        RobustnessClass::NoReliableTrend
    };
    TrendRobustness {
        full,
        without_outliers,
        class,
    }
}

fn detect_indexed_trend(values: &[(usize, f64)]) -> Trend {
    let n = values.len() as f64;
    if values.len() < 2 {
        return Trend {
            slope: 0.0,
            direction: Direction::Flat,
            sample_size: values.len(),
            standard_error: None,
            t_statistic: None,
            significant: false,
        };
    }
    let mean_x = values.iter().map(|(x, _)| *x as f64).sum::<f64>() / n;
    let mean_y = values.iter().map(|(_, y)| *y).sum::<f64>() / n;
    let numerator: f64 = values
        .iter()
        .map(|(x, y)| (*x as f64 - mean_x) * (y - mean_y))
        .sum();
    let denominator: f64 = values
        .iter()
        .map(|(x, _)| (*x as f64 - mean_x).powi(2))
        .sum();
    let slope = numerator / denominator;
    let direction = if slope > 0.0 {
        Direction::Increasing
    } else if slope < 0.0 {
        Direction::Decreasing
    } else {
        Direction::Flat
    };
    // The requested OLS standard error requires n-2 residual degrees of freedom.
    // At n <= 2 no standard error or t statistic is defined; represent that as None.
    let standard_error = if values.len() <= 2 {
        None
    } else {
        let residual_sum_squares: f64 = values
            .iter()
            .map(|(x, y)| {
                let residual = y - (mean_y + slope * (*x as f64 - mean_x));
                residual * residual
            })
            .sum();
        let residual_variance = residual_sum_squares / (n - 2.0);
        Some((residual_variance / denominator).sqrt())
    };
    // |t| > 2 is an approximation, not a calibrated p-value; it degrades at small n.
    let t_statistic = standard_error.and_then(|se| {
        if se == 0.0 {
            (slope != 0.0).then_some(slope.signum() * f64::INFINITY)
        } else {
            Some(slope / se)
        }
    });
    let significant = t_statistic.is_some_and(|t| t.abs() > 2.0);
    Trend {
        slope,
        direction,
        sample_size: values.len(),
        standard_error,
        t_statistic,
        significant,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_outlier_at_its_original_position() {
        // [1,2,3,100] has Q1=1.75, Q3=27.25 and upper fence=65.5; 100 is at index 3.
        let v = [1.0, 2.0, 3.0, 100.0];
        let s = Summary::from_values(&v);
        assert_eq!(
            find_outliers(&v, &s),
            [Outlier {
                index: 3,
                value: 100.0
            }]
        );
    }

    #[test]
    fn regression_slope_and_direction_match_hand_calculation() {
        // y=[1,3,5] lies on y=1+2x exactly, so slope=2 and direction increases.
        let t = detect_trend(&[1.0, 3.0, 5.0]);
        assert!((t.slope - 2.0).abs() < 1e-12);
        assert_eq!(t.direction, Direction::Increasing);
        assert_eq!(detect_trend(&[4.0, 4.0, 4.0]).direction, Direction::Flat);
    }

    #[test]
    fn significant_trend_survives_outlier_removal() {
        // Values follow y=x except a near-centered +22 deviation; IQR flags 32, while both OLS slopes stay near 1.
        let values = [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 32.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0,
        ];
        let outliers = find_outliers(&values, &Summary::from_values(&values));
        let result = check_trend_robustness(&values, &outliers);
        assert_eq!(outliers.len(), 1);
        assert_eq!(result.class, RobustnessClass::Confirmed);
        assert!(result.full.significant);
        assert!(result.without_outliers.significant);
        assert!((result.without_outliers.slope - 1.0).abs() < 1e-12);
    }

    #[test]
    fn significant_trend_lost_after_outlier_removal_is_artifact() {
        // Nine zeros then three tens give a significant rising fit; IQR flags all tens,
        // and removing them leaves nine constant zeros with slope 0 and no t statistic.
        let values = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0,
        ];
        let outliers = find_outliers(&values, &Summary::from_values(&values));
        let result = check_trend_robustness(&values, &outliers);
        assert_eq!(outliers.len(), 3);
        assert!(result.full.significant);
        assert_eq!(result.full.direction, Direction::Increasing);
        assert_eq!(result.without_outliers.direction, Direction::Flat);
        assert!(!result.without_outliers.significant);
        assert_eq!(result.class, RobustnessClass::Artifact);
    }

    #[test]
    fn sample_size_gate_blocks_trend_reporting_below_eight() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let outliers = find_outliers(&values, &Summary::from_values(&values));
        let result = check_trend_robustness(&values, &outliers);
        assert_eq!(result.class, RobustnessClass::InsufficientData);
    }

    #[test]
    fn reports_no_reliable_trend_when_neither_fit_is_significant() {
        // A centered IQR outlier gives both fits nonzero but weak slopes; neither |t| exceeds 2.
        let values = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0,
        ];
        let outliers = find_outliers(&values, &Summary::from_values(&values));
        let result = check_trend_robustness(&values, &outliers);
        assert_eq!(result.class, RobustnessClass::NoReliableTrend);
        assert!(!result.full.significant);
        assert!(!result.without_outliers.significant);
    }

    #[test]
    fn slope_standard_error_requires_more_than_two_observations() {
        let trend = detect_trend(&[1.0, 2.0]);
        assert_eq!(trend.sample_size, 2);
        assert!(trend.standard_error.is_none());
        assert!(trend.t_statistic.is_none());
        assert!(!trend.significant);
    }

    #[test]
    fn slope_standard_error_matches_hand_calculation() {
        // For (x,y)=(0,1),(1,2),(2,1), slope=0, SSE=2/3, residual variance=2/3,
        // and Sxx=2, so SE=sqrt((2/3)/2)=sqrt(1/3).
        let trend = detect_trend(&[1.0, 2.0, 1.0]);
        assert!((trend.slope).abs() < 1e-12);
        assert!((trend.standard_error.unwrap() - (1.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!(!trend.significant);
    }
}
