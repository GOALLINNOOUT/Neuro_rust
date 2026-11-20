use std::env;
use std::fs;
use std::path::PathBuf;

use reason_engine::tokenizer::Tokenizer;
use reason_engine::trainer::{Trainer, TrainingExample};
use reason_engine::transformer::Transformer;

struct Args {
    train: PathBuf,
    validation: PathBuf,
    blimp: PathBuf,
    checkpoint: PathBuf,
    resume: Option<PathBuf>,
    max_stories: usize,
    max_validation_records: usize,
    max_vocab: usize,
    epochs: usize,
    learning_rate: f64,
    dimension: usize,
    heads: usize,
    ff_dim: usize,
    layers: usize,
    sequence_length: usize,
    batch_size: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        train: PathBuf::from("data/tinystories/train.txt"),
        validation: PathBuf::from("data/tinystories/validation.txt"),
        blimp: PathBuf::from("data/blimp/blimp-sample.tsv"),
        checkpoint: PathBuf::from("target/tinystories-stage1.chk"),
        resume: None,
        max_stories: 1000,
        max_validation_records: 1000,
        max_vocab: 4096,
        epochs: 1,
        learning_rate: 0.0001,
        dimension: 64,
        heads: 4,
        ff_dim: 128,
        layers: 2,
        sequence_length: 64,
        batch_size: 8,
    };
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = |values: &mut std::iter::Skip<std::env::Args>, name: &str| {
            values
                .next()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match flag.as_str() {
            "--train" => args.train = PathBuf::from(value(&mut values, "--train")?),
            "--validation" => args.validation = PathBuf::from(value(&mut values, "--validation")?),
            "--blimp" => args.blimp = PathBuf::from(value(&mut values, "--blimp")?),
            "--checkpoint" => args.checkpoint = PathBuf::from(value(&mut values, "--checkpoint")?),
            "--resume" => args.resume = Some(PathBuf::from(value(&mut values, "--resume")?)),
            "--stories" => {
                args.max_stories = value(&mut values, "--stories")?
                    .parse()
                    .map_err(|_| "--stories must be a positive integer")?;
            }
            "--validation-records" => args.max_validation_records = parse_positive(value(&mut values, "--validation-records")?, "validation-records")?,
            "--vocab-size" => {
                args.max_vocab = value(&mut values, "--vocab-size")?
                    .parse()
                    .map_err(|_| "--vocab-size must be at least 5")?;
            }
            "--epochs" => {
                args.epochs = value(&mut values, "--epochs")?
                    .parse()
                    .map_err(|_| "--epochs must be a positive integer")?;
            }
            "--learning-rate" => {
                args.learning_rate = value(&mut values, "--learning-rate")?
                    .parse()
                    .map_err(|_| "--learning-rate must be positive and finite")?;
            }
            "--dimension" => args.dimension = parse_positive(value(&mut values, "--dimension")?, "dimension")?,
            "--heads" => args.heads = parse_positive(value(&mut values, "--heads")?, "heads")?,
            "--ff-dim" => args.ff_dim = parse_positive(value(&mut values, "--ff-dim")?, "ff-dim")?,
            "--layers" => args.layers = parse_positive(value(&mut values, "--layers")?, "layers")?,
            "--sequence-length" => args.sequence_length = parse_positive(value(&mut values, "--sequence-length")?, "sequence-length")?,
            "--batch-size" => args.batch_size = parse_positive(value(&mut values, "--batch-size")?, "batch-size")?,
            "--help" | "-h" => {
                println!("tinystories-stage1 [--train PATH] [--validation PATH] [--blimp PATH] [--checkpoint PATH] [--resume PATH] [--stories N] [--validation-records N] [--vocab-size N] [--dimension N] [--heads N] [--ff-dim N] [--layers N] [--sequence-length N] [--batch-size N] [--epochs N] [--learning-rate RATE]");
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }
    if args.max_stories == 0 || args.max_vocab < 5 || args.epochs == 0 || args.dimension == 0 || args.heads == 0 || args.ff_dim == 0 || args.layers == 0 || args.sequence_length == 0 || args.batch_size == 0 {
        return Err(
            "stories, epochs, and vocabulary size must be positive (vocabulary >= 5)".into(),
        );
    }
    if !args.dimension.is_multiple_of(args.heads) {
        return Err("dimension must be divisible by heads".into());
    }
    if !args.learning_rate.is_finite() || args.learning_rate <= 0.0 {
        return Err("learning rate must be positive and finite".into());
    }
    Ok(args)
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let mut train_stories = read_stories(&args.train)?;
    let mut validation_stories = read_stories(&args.validation)?;
    if train_stories.is_empty() || validation_stories.is_empty() {
        return Err("training and validation files must both contain stories".into());
    }
    let corpus_refs: Vec<&str> = train_stories.iter().map(String::as_str).collect();
    let fresh_tokenizer = Tokenizer::build(&corpus_refs, args.max_vocab);
    train_stories = evenly_spaced_sample(train_stories, args.max_stories);
    validation_stories = evenly_spaced_sample(validation_stories, args.max_validation_records);

    let (mut model, tokenizer) = if let Some(path) = &args.resume {
        let (model, tokenizer) = Transformer::load_checkpoint(path)?;
        if model.max_seq_len() != args.sequence_length || model.embedding_dim() != args.dimension || model.num_heads() != args.heads || model.ff_dim() != args.ff_dim || model.num_layers() != args.layers {
            return Err(format!(
                "checkpoint architecture is d_model={}, heads={}, ff_dim={}, layers={}, context={}; requested d_model={}, heads={}, ff_dim={}, layers={}, context={}",
                model.embedding_dim(), model.num_heads(), model.ff_dim(), model.num_layers(), model.max_seq_len(),
                args.dimension, args.heads, args.ff_dim, args.layers, args.sequence_length
            ));
        }
        (model, tokenizer)
    } else {
        let model = Transformer::new(fresh_tokenizer.vocab_size(), args.dimension, args.heads, args.ff_dim, args.layers, args.sequence_length, 42)?;
        (model, fresh_tokenizer)
    };
    let training = make_examples(&tokenizer, &train_stories, args.sequence_length);
    let validation = make_examples(&tokenizer, &validation_stories, args.sequence_length);
    if training.is_empty() || validation.is_empty() {
        return Err("no usable training or validation sequences were produced".into());
    }
    let training_oov = unknown_rate(&training, tokenizer.unk_id());
    let validation_oov = unknown_rate(&validation, tokenizer.unk_id());

    let before_validation = mean_loss(&model, &validation)?;
    println!("Stage 1: next-token language modeling");
    println!("Training records: {} | validation records: {} | training chunks: {} | validation chunks: {}", train_stories.len(), validation_stories.len(), training.len(), validation.len());
    println!("Architecture: d_model={} | heads={} | FFN={} | layers={} | context={} | batch={} ({} optimizer updates/epoch)", args.dimension, args.heads, args.ff_dim, args.layers, args.sequence_length, args.batch_size, training.len().div_ceil(args.batch_size));
    println!("{}", tokenizer.build_summary());
    println!(
        "Unknown-token rate: training {:.2}%; validation {:.2}%",
        training_oov * 100.0,
        validation_oov * 100.0
    );
    println!("Validation loss before this run: {before_validation:.4}");
    println!(
        "Training up to {} epoch(s), learning rate {}...",
        args.epochs, args.learning_rate
    );

    if let Some(parent) = args
        .checkpoint
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create checkpoint directory: {error}"))?;
    }
    let mut best_model = model.clone();
    let mut best_validation = before_validation;
    if args.resume.is_none() {
        model.save_checkpoint(&tokenizer, &args.checkpoint)?;
    }
    for epoch in 1..=args.epochs {
        let curve = Trainer::train_minibatch(&mut model, &training, 1, args.learning_rate, args.batch_size, 0)?;
        let validation_loss = mean_loss(&model, &validation)?;
        println!(
            "Epoch {epoch}: train loss {:.4}; validation loss {:.4}",
            curve[0], validation_loss
        );
        if validation_loss > best_validation {
            println!("Validation worsened ({best_validation:.4} -> {validation_loss:.4}); stopping and keeping the best checkpoint.");
            break;
        }
        best_validation = validation_loss;
        best_model = model.clone();
        model.save_checkpoint(&tokenizer, &args.checkpoint)?;
        println!("Saved improved checkpoint to {}", args.checkpoint.display());
    }
    model = best_model;
    println!("Best validation loss: {best_validation:.4}");
    if args.blimp.exists() {
        evaluate_blimp(&model, &tokenizer, &args.blimp)?;
    }
    Ok(())
}

fn evenly_spaced_sample(records: Vec<String>, limit: usize) -> Vec<String> {
    if records.len() <= limit { return records; }
    if limit == 1 { return vec![records[records.len() / 2].clone()]; }
    let last = records.len() - 1;
    (0..limit)
        .map(|index| records[index * last / (limit - 1)].clone())
        .collect()
}

fn parse_positive(value: String, name: &str) -> Result<usize, String> {
    let number = value.parse::<usize>().map_err(|_| format!("--{name} must be a positive integer"))?;
    if number == 0 { return Err(format!("--{name} must be greater than zero")); }
    Ok(number)
}

fn read_stories(path: &PathBuf) -> Result<Vec<String>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read stories '{}': {error}", path.display()))?;
    Ok(text
        .split("\n\n")
        .map(|story| story.replace('\n', " ").trim().to_string())
        .filter(|story| !story.is_empty())
        .collect())
}

fn make_examples(
    tokenizer: &Tokenizer,
    stories: &[String],
    max_seq_len: usize,
) -> Vec<TrainingExample> {
    let body_len = max_seq_len - 2;
    let mut examples = Vec::new();
    for story in stories {
        let (ids, numbers) = tokenizer.tokenize(story);
        let body_ids = &ids[1..ids.len() - 1];
        let body_numbers = &numbers[1..numbers.len() - 1];
        for start in (0..body_ids.len()).step_by(body_len) {
            let end = (start + body_len).min(body_ids.len());
            let mut chunk_ids = Vec::with_capacity(end - start + 2);
            let mut chunk_numbers = Vec::with_capacity(end - start + 2);
            chunk_ids.push(tokenizer.bos_id());
            chunk_numbers.push(None);
            chunk_ids.extend_from_slice(&body_ids[start..end]);
            chunk_numbers.extend_from_slice(&body_numbers[start..end]);
            chunk_ids.push(tokenizer.eos_id());
            chunk_numbers.push(None);
            if chunk_ids.len() >= 3 {
                examples.push(TrainingExample {
                    token_ids: chunk_ids,
                    numeric_values: chunk_numbers,
                });
            }
        }
    }
    examples
}

fn mean_loss(model: &Transformer, examples: &[TrainingExample]) -> Result<f64, String> {
    let mut weighted_loss = 0.0;
    let mut target_count = 0usize;
    for example in examples {
        let logits = model.forward(&example.token_ids, &example.numeric_values)?;
        let count = example.token_ids.len() - 1;
        weighted_loss += model.loss(&logits, &example.token_ids) * count as f64;
        target_count += count;
    }
    Ok(weighted_loss / target_count as f64)
}

fn unknown_rate(examples: &[TrainingExample], unk_id: usize) -> f64 {
    let total = examples
        .iter()
        .map(|example| example.token_ids.len())
        .sum::<usize>();
    if total == 0 {
        return 0.0;
    }
    let unknown = examples
        .iter()
        .flat_map(|example| &example.token_ids)
        .filter(|&&id| id == unk_id)
        .count();
    unknown as f64 / total as f64
}

fn evaluate_blimp(
    model: &Transformer,
    tokenizer: &Tokenizer,
    path: &PathBuf,
) -> Result<(), String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read BLiMP sample '{}': {error}", path.display()))?;
    let mut correct = 0usize;
    let mut skipped_unknown = 0usize;
    let mut seen = 0usize;
    for (line_number, line) in text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 3 {
            return Err(format!(
                "BLiMP TSV line {} must have three columns",
                line_number + 1
            ));
        }
        let good = tokenizer.tokenize(fields[1]);
        let bad = tokenizer.tokenize(fields[2]);
        if good.0.contains(&tokenizer.unk_id()) || bad.0.contains(&tokenizer.unk_id()) {
            skipped_unknown += 1;
            continue;
        }
        let good_logits = model.forward(&good.0, &good.1)?;
        let bad_logits = model.forward(&bad.0, &bad.1)?;
        let good_loss = model.loss(&good_logits, &good.0) * (good.0.len() - 1) as f64;
        let bad_loss = model.loss(&bad_logits, &bad.0) * (bad.0.len() - 1) as f64;
        correct += usize::from(good_loss < bad_loss);
        seen += 1;
    }
    println!("BLiMP sample: {correct}/{seen} correct ({:.1}%); skipped {skipped_unknown} pairs with out-of-vocabulary words", if seen == 0 { 0.0 } else { 100.0 * correct as f64 / seen as f64 });
    println!("This is a small sample of BLiMP, not the full 67,000-pair benchmark.");
    Ok(())
}
