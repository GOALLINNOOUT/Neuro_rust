#[derive(Debug, Clone)]
pub struct CsvScenario {
    pub trend: f64,
    pub noise: f64,
    pub outlier_positions: Vec<usize>,
    pub outlier_magnitude: f64,
    pub base_value: f64,
}

/// Builds a reproducible 25-row CSV for one synthetic analytical scenario.
pub fn generate_csv(scenario: &CsvScenario, seed: u64) -> String {
    let mut csv = String::from("index,measurement\n");
    for index in 0..25 {
        let mut rng = seed.wrapping_add(index as u64);
        let mut measurement = scenario.base_value
            + scenario.trend * index as f64
            + gaussian(scenario.noise, &mut rng);
        if scenario.outlier_positions.contains(&index) {
            measurement += scenario.outlier_magnitude;
        }
        csv.push_str(&format!("{index},{measurement:.6}\n"));
    }
    csv
}

/// Returns the ten Phase 2 scenarios paired with their fixed seeds.
pub fn scenarios() -> Vec<(CsvScenario, u64)> {
    vec![
        (scenario(2.0, 0.5, vec![], 0.0, 10.0), 1),
        (scenario(-2.0, 0.5, vec![], 0.0, 50.0), 2),
        (scenario(0.0, 0.5, vec![], 0.0, 20.0), 3),
        (scenario(0.0, 8.0, vec![], 0.0, 20.0), 4),
        (scenario(0.5, 4.0, vec![], 0.0, 15.0), 5),
        (scenario(-0.5, 4.0, vec![], 0.0, 35.0), 6),
        (scenario(2.0, 0.5, vec![12], 30.0, 10.0), 7),
        (scenario(2.0, 0.5, vec![5, 12, 19], 25.0, 10.0), 8),
        (scenario(0.0, 1.0, vec![8, 9, 10], 20.0, 20.0), 9),
        (scenario(0.0, 0.05, vec![], 0.0, 100.0), 10),
    ]
}

fn scenario(
    trend: f64,
    noise: f64,
    outlier_positions: Vec<usize>,
    outlier_magnitude: f64,
    base_value: f64,
) -> CsvScenario {
    CsvScenario {
        trend,
        noise,
        outlier_positions,
        outlier_magnitude,
        base_value,
    }
}

fn gaussian(std_dev: f64, rng: &mut u64) -> f64 {
    let mut u1 = splitmix64(rng);
    if u1 == 0.0 {
        u1 = f64::MIN_POSITIVE;
    }
    let u2 = splitmix64(rng);
    std_dev * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn splitmix64(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
}

#[cfg(test)]
mod tests {
    use super::{generate_csv, scenarios};
    use std::fs;
    use std::path::Path;

    #[test]
    fn generated_csvs_are_valid() {
        let output_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/generated");
        fs::create_dir_all(&output_dir).unwrap();

        for (index, (scenario, seed)) in scenarios().iter().enumerate() {
            let csv = generate_csv(scenario, *seed);
            let lines: Vec<_> = csv.lines().collect();
            assert_eq!(lines[0], "index,measurement");
            assert_eq!(lines.len() - 1, 25);
            for row in &lines[1..] {
                let measurement = row.split(',').nth(1).unwrap().parse::<f64>().unwrap();
                assert!(measurement.is_finite(), "non-finite measurement in {row}");
            }

            println!("scenario_{:02}.csv first rows:", index + 1);
            for row in lines.iter().take(3) {
                println!("{row}");
            }
            let path = output_dir.join(format!("scenario_{:02}.csv", index + 1));
            fs::write(path, csv).unwrap();
        }
    }
}
