use std::env;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use reason_engine::tokenizer::Tokenizer;
use reason_engine::transformer::Transformer;

fn main() -> Result<(), String> {
    let mut checkpoint = PathBuf::from("target/tinystories-stage1.chk");
    let mut max_new_tokens = 40usize;
    let mut greedy = false;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--checkpoint" => {
                checkpoint = PathBuf::from(values.next().ok_or("--checkpoint requires a path")?);
            }
            "--tokens" => {
                max_new_tokens = values
                    .next()
                    .ok_or("--tokens requires a number")?
                    .parse()
                    .map_err(|_| "--tokens must be a positive integer")?;
            }
            "--greedy" => greedy = true,
            "--help" | "-h" => {
                println!("experience [--checkpoint PATH] [--tokens N] [--greedy]");
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if max_new_tokens == 0 {
        return Err("--tokens must be greater than zero".into());
    }

    let (model, tokenizer) = Transformer::load_checkpoint(&checkpoint)?;
    println!(
        "NERO Rust text completion (vocabulary: {} words)",
        tokenizer.vocab_size()
    );
    println!("Type any prompt. The model continues it; it does not reliably answer questions yet. Type /quit to exit.");
    let stdin = io::stdin();
    let mut rng_state = 42u64;
    loop {
        print!("\nYou> ");
        io::stdout().flush().map_err(|error| error.to_string())?;
        let mut line = String::new();
        if stdin
            .lock()
            .read_line(&mut line)
            .map_err(|error| error.to_string())?
            == 0
        {
            break;
        }
        let prompt = line.trim();
        if prompt.eq_ignore_ascii_case("/quit") || prompt.eq_ignore_ascii_case("/exit") {
            break;
        }
        if prompt.is_empty() {
            continue;
        }
        let (mut ids, mut numeric_values) = tokenizer.tokenize(prompt);
        ids.pop();
        numeric_values.pop();
        let unknown_words = ids.iter().filter(|&&id| id == tokenizer.unk_id()).count();
        println!(
            "[{} prompt word(s) are outside the vocabulary]",
            unknown_words
        );
        let output = complete(
            &model,
            &tokenizer,
            &mut ids,
            &mut numeric_values,
            max_new_tokens,
            &mut rng_state,
            greedy,
        )?;
        println!("Model> {output}");
    }
    Ok(())
}

fn complete(
    model: &Transformer,
    tokenizer: &Tokenizer,
    ids: &mut Vec<usize>,
    numbers: &mut Vec<Option<f64>>,
    max_new_tokens: usize,
    rng_state: &mut u64,
    greedy: bool,
) -> Result<String, String> {
    let prompt_ids = ids.clone();
    let mut generated_ids = Vec::new();
    for _ in 0..max_new_tokens {
        let max_len = model.max_seq_len();
        if ids.len() > max_len {
            let keep_from = ids.len() - (max_len - 1);
            ids.drain(1..keep_from);
            numbers.drain(1..keep_from);
        }
        let logits = model.forward(ids, numbers)?;
        let last_logits = logits.last().ok_or("model returned no logits")?;
        let next = if greedy {
            greedy_next(last_logits, tokenizer)
        } else {
            sample_top_k(last_logits, tokenizer, rng_state)
        };
        if next == tokenizer.eos_id() {
            break;
        }
        ids.push(next);
        numbers.push(None);
        generated_ids.push(next);
    }
    let mut complete_ids = prompt_ids;
    complete_ids.extend(generated_ids);
    Ok(tokenizer.detokenize(&complete_ids))
}

fn greedy_next(logits: &[f64], tokenizer: &Tokenizer) -> usize {
    logits
        .iter()
        .enumerate()
        .filter(|(id, value)| {
            *id != tokenizer.bos_id()
                && *id != tokenizer.pad_id()
                && *id != tokenizer.unk_id()
                && value.is_finite()
        })
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(id, _)| id)
        .unwrap_or(tokenizer.eos_id())
}

fn sample_top_k(logits: &[f64], tokenizer: &Tokenizer, rng_state: &mut u64) -> usize {
    let mut ranked: Vec<(usize, f64)> = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(id, value)| {
            *id != tokenizer.bos_id()
                && *id != tokenizer.pad_id()
                && *id != tokenizer.unk_id()
                && value.is_finite()
        })
        .collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    ranked.truncate(12);
    let Some((best_id, best_logit)) = ranked.first().copied() else {
        return tokenizer.eos_id();
    };
    let weights: Vec<f64> = ranked
        .iter()
        .map(|(_, logit)| ((logit - best_logit) / 0.8).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return best_id;
    }
    *rng_state ^= *rng_state << 13;
    *rng_state ^= *rng_state >> 7;
    *rng_state ^= *rng_state << 17;
    let mut choice = (*rng_state as f64 / u64::MAX as f64) * total;
    for ((token_id, _), weight) in ranked.iter().zip(weights) {
        if choice < weight {
            return *token_id;
        }
        choice -= weight;
    }
    best_id
}
