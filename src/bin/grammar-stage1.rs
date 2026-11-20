use std::env;
use std::fs;
use std::path::PathBuf;

use reason_engine::tokenizer::Tokenizer;
use reason_engine::trainer::{Trainer, TrainingExample};
use reason_engine::transformer::Transformer;

const TRAIN: &str = include_str!("../../fixtures/grammar_stage1_train.txt");
const EVAL: &str = include_str!("../../fixtures/grammar_stage1_eval.tsv");

struct Args {
    epochs: usize,
    learning_rate: f64,
    resume: Option<PathBuf>,
    checkpoint: PathBuf,
    eval: Option<PathBuf>,
    guidance: Option<PathBuf>,
    eval_only: bool,
}

fn args() -> Result<Args, String> {
    let mut parsed = Args {
        epochs: 400,
        learning_rate: 0.01,
        resume: None,
        checkpoint: PathBuf::from("target/grammar-stage1.chk"),
        eval: None,
        guidance: None,
        eval_only: false,
    };
    let mut values = env::args().skip(1);
    while let Some(arg) = values.next() {
        match arg.as_str() {
            "--epochs" => {
                parsed.epochs = values
                    .next()
                    .ok_or("--epochs requires a number")?
                    .parse()
                    .map_err(|_| "--epochs must be a positive integer")?;
            }
            "--learning-rate" => {
                parsed.learning_rate = values
                    .next()
                    .ok_or("--learning-rate requires a number")?
                    .parse()
                    .map_err(|_| "--learning-rate must be a positive finite number")?;
            }
            "--resume" => {
                parsed.resume = Some(PathBuf::from(
                    values.next().ok_or("--resume requires a checkpoint path")?,
                ));
            }
            "--checkpoint" => {
                parsed.checkpoint =
                    PathBuf::from(values.next().ok_or("--checkpoint requires a path")?);
            }
            "--eval" => {
                parsed.eval = Some(PathBuf::from(
                    values.next().ok_or("--eval requires a TSV path")?,
                ));
            }
            "--guidance" => {
                parsed.guidance = Some(PathBuf::from(
                    values.next().ok_or("--guidance requires a text path")?,
                ));
            }
            "--eval-only" => parsed.eval_only = true,
            "--help" | "-h" => {
                println!("grammar-stage1 [--epochs N] [--learning-rate RATE] [--resume PATH] [--checkpoint PATH] [--eval PATH] [--guidance PATH] [--eval-only]");
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }
    if parsed.epochs == 0 {
        return Err("--epochs must be greater than zero".into());
    }
    if !parsed.learning_rate.is_finite() || parsed.learning_rate <= 0.0 {
        return Err("--learning-rate must be a positive finite number".into());
    }
    if parsed.eval_only && parsed.resume.is_none() {
        return Err("--eval-only requires --resume PATH".into());
    }
    if parsed.guidance.is_some() && parsed.resume.is_none() {
        return Err(
            "--guidance requires --resume PATH so the existing vocabulary is preserved".into(),
        );
    }
    Ok(parsed)
}

fn main() -> Result<(), String> {
    let args = args()?;
    let training_text = if let Some(path) = &args.guidance {
        fs::read_to_string(path)
            .map_err(|error| format!("cannot read guidance file '{}': {error}", path.display()))?
    } else {
        TRAIN.to_string()
    };
    let corpus: Vec<&str> = training_text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    let eval_text = if let Some(path) = &args.eval {
        fs::read_to_string(path)
            .map_err(|error| format!("cannot read eval file '{}': {error}", path.display()))?
    } else {
        EVAL.to_string()
    };
    let (mut model, tokenizer, starting_epoch) = if let Some(path) = &args.resume {
        let (model, tokenizer) = Transformer::load_checkpoint(path)?;
        (model, tokenizer, "resumed")
    } else {
        let tokenizer = Tokenizer::build(&corpus, 512);
        let model = Transformer::new(tokenizer.vocab_size(), 32, 2, 64, 1, 32, 42)?;
        (model, tokenizer, "new")
    };

    let examples: Vec<_> = corpus
        .iter()
        .map(|sentence| TrainingExample::from_text(&tokenizer, sentence))
        .collect();
    let pairs = parse_eval(&eval_text, &tokenizer)?;
    let unknown = examples
        .iter()
        .flat_map(|example| example.token_ids.iter())
        .chain(pairs.iter().flat_map(|pair| pair.grammatical.0.iter()))
        .chain(pairs.iter().flat_map(|pair| pair.ungrammatical.0.iter()))
        .filter(|&&id| id == tokenizer.unk_id())
        .count();
    if unknown > 0 {
        return Err(format!(
            "stage data contains {unknown} unknown tokens; expand vocabulary before continuing"
        ));
    }

    println!("Grammar Stage 1: basic sentence structure");
    println!("Training sentences: {} | Held-out pairs: {} | Vocabulary: {} | Starting state: {starting_epoch}", if args.eval_only { 0 } else { examples.len() }, pairs.len(), tokenizer.vocab_size());
    if args.eval_only {
        println!("Training skipped; evaluating the saved checkpoint.");
    } else {
        println!(
            "Training for {} epochs at learning rate {}...",
            args.epochs, args.learning_rate
        );
        let curve = Trainer::train(&mut model, &examples, args.epochs, args.learning_rate, 0)?;
        println!(
            "Mean training loss: {:.4} -> {:.4}",
            curve.first().unwrap(),
            curve.last().unwrap()
        );
    }

    if let Some(parent) = args
        .checkpoint
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create checkpoint directory: {error}"))?;
    }
    let scored_model = if args.eval_only {
        model
    } else {
        let probe = tokenizer.tokenize("the curious cat sleeps");
        let before_logits = model.forward(&probe.0, &probe.1)?;
        model.save_checkpoint(&tokenizer, &args.checkpoint)?;
        let (loaded_model, loaded_tokenizer) = Transformer::load_checkpoint(&args.checkpoint)?;
        let loaded_probe = loaded_tokenizer.tokenize("the curious cat sleeps");
        if probe.0 != loaded_probe.0 || probe.1 != loaded_probe.1 {
            return Err("checkpoint round-trip changed tokenizer output".into());
        }
        let after_logits = loaded_model.forward(&loaded_probe.0, &loaded_probe.1)?;
        let max_logit_delta = before_logits
            .iter()
            .flatten()
            .zip(after_logits.iter().flatten())
            .map(|(before, after)| (before - after).abs())
            .fold(0.0_f64, f64::max);
        if max_logit_delta > 1e-12 {
            return Err(format!(
                "checkpoint round-trip changed logits (max delta {max_logit_delta})"
            ));
        }
        println!(
            "Checkpoint round-trip: exact tokenizer output; max logit delta {max_logit_delta:.3e}"
        );
        println!("Checkpoint saved: {}", args.checkpoint.display());
        loaded_model
    };
    report_scores(&scored_model, &pairs)?;
    Ok(())
}

struct Pair {
    category: String,
    grammatical: (Vec<usize>, Vec<Option<f64>>),
    ungrammatical: (Vec<usize>, Vec<Option<f64>>),
}

fn parse_eval(text: &str, tokenizer: &Tokenizer) -> Result<Vec<Pair>, String> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(line_index, line)| {
            let columns: Vec<_> = line.split('\t').collect();
            if columns.len() != 3 {
                return Err(format!(
                    "eval line {} must have three tab-separated columns",
                    line_index + 1
                ));
            }
            Ok(Pair {
                category: columns[0].to_string(),
                grammatical: tokenizer.tokenize(columns[1]),
                ungrammatical: tokenizer.tokenize(columns[2]),
            })
        })
        .collect()
}

fn report_scores(model: &Transformer, pairs: &[Pair]) -> Result<(), String> {
    let mut category_totals: Vec<(String, usize, usize, f64)> = Vec::new();
    let mut wins = 0;
    for pair in pairs {
        let good_logits = model.forward(&pair.grammatical.0, &pair.grammatical.1)?;
        let bad_logits = model.forward(&pair.ungrammatical.0, &pair.ungrammatical.1)?;
        let good_loss = model.loss(&good_logits, &pair.grammatical.0);
        let bad_loss = model.loss(&bad_logits, &pair.ungrammatical.0);
        let chosen_good = good_loss < bad_loss;
        wins += usize::from(chosen_good);
        let margin = bad_loss - good_loss;
        let category = category_totals
            .iter_mut()
            .find(|(name, _, _, _)| name == &pair.category);
        if let Some((_, correct, total, margin_sum)) = category {
            *correct += usize::from(chosen_good);
            *total += 1;
            *margin_sum += margin;
        } else {
            category_totals.push((pair.category.clone(), usize::from(chosen_good), 1, margin));
        }
    }
    println!(
        "Held-out grammatical preference: {wins}/{} ({:.1}%; chance baseline 50%)",
        pairs.len(),
        100.0 * wins as f64 / pairs.len() as f64
    );
    for (category, correct, total, margin_sum) in category_totals {
        println!(
            "  {category}: {correct}/{total} (mean loss margin {:+.4})",
            margin_sum / total as f64
        );
    }
    println!("Scoring: lower mean next-token loss is preferred; this small diagnostic does not establish general language understanding.");
    Ok(())
}
