mod data;
pub mod corpus;
mod explain;
mod patterns;
mod stats;

use std::env;
use std::process::ExitCode;

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.get(1).is_some_and(|arg| arg == "--corpus") {
        if args.len() != 5 {
            return Err(format!(
                "Usage: {} --corpus <csv-file> <column-index-or-header> <output-file>",
                args.first().map_or("reason-engine", String::as_str)
            ));
        }
        let sentences = corpus::generate(&args[2], &args[3])?;
        corpus::write(&args[4], &sentences)?;
        for sentence in sentences {
            println!("{sentence}");
        }
        return Ok(());
    }
    if args.len() != 3 {
        return Err(format!(
            "Usage: {} <csv-file> <column-index-or-header>",
            args.first().map_or("reason-engine", String::as_str)
        ));
    }
    let values = data::load_column(&args[1], &args[2])?;
    let summary = stats::Summary::from_values(&values);
    let outliers = patterns::find_outliers(&values, &summary);
    let robustness = patterns::check_trend_robustness(&values, &outliers);
    println!("{}", explain::generate(&summary, &outliers, &robustness));
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("Error: {message}");
            ExitCode::FAILURE
        }
    }
}
