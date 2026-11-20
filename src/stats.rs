#[derive(Debug, Clone, Copy)]
pub struct Summary {
    pub mean: f64,
    pub variance: f64,
    pub standard_deviation: f64,
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
    pub iqr: f64,
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Unbiased sample variance (denominator n - 1); a singleton has zero variance.
pub fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        return 0.0;
    }
    let average = mean(values);
    values.iter().map(|x| (x - average).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

pub fn standard_deviation(values: &[f64]) -> f64 {
    variance(values).sqrt()
}

/// Quantile using the common linear interpolation rule with h = (n - 1) * p.
pub fn quantile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
}

pub fn median(values: &[f64]) -> f64 {
    quantile(values, 0.5)
}
pub fn q1(values: &[f64]) -> f64 {
    quantile(values, 0.25)
}
pub fn q3(values: &[f64]) -> f64 {
    quantile(values, 0.75)
}
pub fn iqr(values: &[f64]) -> f64 {
    q3(values) - q1(values)
}

impl Summary {
    pub fn from_values(values: &[f64]) -> Self {
        let q1 = q1(values);
        let q3 = q3(values);
        Self {
            mean: mean(values),
            variance: variance(values),
            standard_deviation: standard_deviation(values),
            median: median(values),
            q1,
            q3,
            iqr: iqr(values),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statistics_match_hand_calculated_values() {
        // For [1,2,3,4], mean/median are 2.5; sample variance is 5/3.
        // Linear interpolation at h=0.75 and 2.25 gives Q1=1.75, Q3=3.25, IQR=1.5.
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(mean(&v), 2.5);
        assert_eq!(median(&v), 2.5);
        assert!((variance(&v) - 5.0 / 3.0).abs() < 1e-12);
        assert!((standard_deviation(&v) - (5.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(q1(&v), 1.75);
        assert_eq!(q3(&v), 3.25);
        assert_eq!(iqr(&v), 1.5);
    }

    #[test]
    fn handles_empty_single_and_identical_inputs() {
        assert!(mean(&[]).is_nan());
        assert_eq!(variance(&[7.0]), 0.0);
        assert_eq!(standard_deviation(&[7.0, 7.0]), 0.0);
        assert_eq!(median(&[7.0]), 7.0);
    }
}
