use reason_engine::tokenizer::Tokenizer;
use reason_engine::trainer::{Trainer, TrainingExample};
use reason_engine::transformer::Transformer;
use std::io::{self, Write};
use std::time::Instant;

fn main() -> Result<(), String> {
    let corpus_path = format!("{}/fixtures/corpus_v2.txt", env!("CARGO_MANIFEST_DIR"));
    let corpus_text = std::fs::read_to_string(corpus_path)
        .map_err(|error| format!("could not read corpus: {error}"))?;
    let sentences: Vec<&str> = corpus_text.lines().collect();
    let tokenizer = Tokenizer::build(&sentences, 2048);
    let examples: Vec<_> = sentences
        .iter()
        .map(|sentence| TrainingExample::from_text(&tokenizer, sentence))
        .collect();
    let mut model = Transformer::new(tokenizer.vocab_size(), 128, 4, 512, 4, 64, 42)?;
    let average_loss = |model: &Transformer| -> f64 {
        examples
            .iter()
            .map(|example| {
                model
                    .loss(
                        &model
                            .forward(&example.token_ids, &example.numeric_values)
                            .unwrap(),
                        &example.token_ids,
                    )
            })
            .sum::<f64>()
            / examples.len() as f64
    };

    let training_started = Instant::now();
    for epoch in 1..=1000 {
        Trainer::train(&mut model, &examples, 1, 0.1, 0)?;
        let elapsed = training_started.elapsed().as_secs_f64();
        let estimated_remaining = elapsed / epoch as f64 * (1000 - epoch) as f64;
        println!(
            "Progress: epoch {epoch}/1000 ({:.1}%), elapsed {:.1}s, estimated remaining {:.1}s",
            epoch as f64 / 10.0,
            elapsed,
            estimated_remaining
        );
        io::stdout().flush().map_err(|error| error.to_string())?;

        if epoch % 200 == 0 || epoch == 100 {
            println!(
                "Phase 2 corpus loss epoch {epoch}: {:.6}",
                average_loss(&model)
            );
            io::stdout().flush().map_err(|error| error.to_string())?;
        }
    }

    for prompt_text in [
        "The standard is significant",
        "Neither variance remains reliable",
        "These observations show increasing trend",
        "The mean is",
        "The fit is",
    ] {
        let (mut prompt, _) = tokenizer.tokenize(prompt_text);
        prompt.pop();
        let generated = model.generate(&prompt, 20)?;
        println!("Prompt: {prompt_text}");
        println!("Generated token IDs: {generated:?}");
        println!("Generated text: {}", tokenizer.detokenize(&generated));
    }

    Ok(())
}
