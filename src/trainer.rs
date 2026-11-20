use crate::tokenizer::Tokenizer;
use crate::transformer::Transformer;
use std::sync::atomic::{AtomicU64, Ordering};

static LAST_EPOCH_MAX_GRADIENT: AtomicU64 = AtomicU64::new(0);
static DIAGNOSTICS_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn diagnostic_last_epoch_max_gradient() -> f64 { f64::from_bits(LAST_EPOCH_MAX_GRADIENT.load(Ordering::Relaxed)) }
pub fn set_diagnostics_enabled(enabled: bool) { DIAGNOSTICS_ENABLED.store(enabled, Ordering::Relaxed); }

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingExample {
    pub token_ids: Vec<usize>,
    pub numeric_values: Vec<Option<f64>>,
}

impl TrainingExample {
    pub fn from_text(tokenizer: &Tokenizer, text: &str) -> Self {
        let (token_ids, numeric_values) = tokenizer.tokenize(text);
        Self { token_ids, numeric_values }
    }
}

pub struct Trainer;

impl Trainer {
    /// Online SGD: one optimizer update follows each example. Returns the mean
    /// pre-update loss for each epoch. Set `print_every` to zero to silence logs.
    pub fn train(model: &mut Transformer, examples: &[TrainingExample], epochs: usize, learning_rate: f64, print_every: usize) -> Result<Vec<f64>, String> {
        if examples.is_empty() { return Err("training examples must not be empty".into()); }
        let mut epoch_losses = Vec::with_capacity(epochs);
        let mut step = 0usize;
        for epoch in 0..epochs {
            let mut total = 0.0;
            let mut max_gradient = 0.0_f64;
            for (index, example) in examples.iter().enumerate() {
                let loss = model.backward(&example.token_ids, &example.numeric_values)
                    .map_err(|error| format!("training example {} failed: {error}", index + 1))?;
                if DIAGNOSTICS_ENABLED.load(Ordering::Relaxed) {
                    max_gradient = max_gradient.max(model.diagnostic_extrema().iter().map(|(_, _, grad)| *grad).fold(0.0, f64::max));
                }
                model.apply_gradients(learning_rate)?;
                total += loss;
                step += 1;
                if print_every > 0 && step.is_multiple_of(print_every) { println!("Epoch {}, Example {}, Loss: {:.4}", epoch + 1, index + 1, loss); }
            }
            if DIAGNOSTICS_ENABLED.load(Ordering::Relaxed) { LAST_EPOCH_MAX_GRADIENT.store(max_gradient.to_bits(), Ordering::Relaxed); }
            epoch_losses.push(total / examples.len() as f64);
        }
        Ok(epoch_losses)
    }

    /// Accumulates gradients over a small batch and applies one averaged SGD
    /// update. This avoids sweeping the full parameter matrices after every
    /// example, which is especially costly for the vocabulary projection.
    pub fn train_minibatch(model: &mut Transformer, examples: &[TrainingExample], epochs: usize, learning_rate: f64, batch_size: usize, print_every: usize) -> Result<Vec<f64>, String> {
        if examples.is_empty() { return Err("training examples must not be empty".into()); }
        if batch_size == 0 { return Err("batch_size must be greater than zero".into()); }
        let mut epoch_losses = Vec::with_capacity(epochs);
        let mut step = 0usize;
        for epoch in 0..epochs {
            let mut total = 0.0;
            for (index, example) in examples.iter().enumerate() {
                total += model.backward(&example.token_ids, &example.numeric_values)?;
                step += 1;
                let batch_count = (index % batch_size) + 1;
                let end_of_batch = batch_count == batch_size || index + 1 == examples.len();
                if end_of_batch {
                    model.apply_gradients(learning_rate / batch_count as f64)?;
                }
                if print_every > 0 && step.is_multiple_of(print_every) { println!("Epoch {}, Example {}, Loss: {:.4}", epoch + 1, index + 1, total / (index + 1) as f64); }
            }
            epoch_losses.push(total / examples.len() as f64);
        }
        Ok(epoch_losses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Tokenizer;
    use crate::transformer::Transformer;

    #[test]
    fn overfits_five_analytical_sentences_in_two_hundred_epochs() {
        const CORPUS: [&str; 5] = [
            "The trend is increasing",
            "The trend is decreasing",
            "The mean is stable",
            "An outlier changes the trend",
            "The trend remains significant",
        ];
        let tokenizer = Tokenizer::build(&CORPUS, 2048);
        let examples: Vec<_> = CORPUS.iter().map(|s| TrainingExample::from_text(&tokenizer, s)).collect();
        let mut model = Transformer::new(tokenizer.vocab_size(), 32, 2, 64, 1, 64, 42).unwrap();
        let average_loss = |model: &Transformer| -> f64 {
            examples.iter().map(|e| model.loss(&model.forward(&e.token_ids, &e.numeric_values).unwrap(), &e.token_ids)).sum::<f64>() / examples.len() as f64
        };
        let mut curve = vec![(0, average_loss(&model))];
        for epoch in (20..=200).step_by(20) {
            Trainer::train(&mut model, &examples, 20, 0.1, 0).unwrap();
            curve.push((epoch, average_loss(&model)));
        }
        for (epoch, loss) in &curve { println!("Overfit loss epoch {epoch}: {loss:.6}"); }
        let final_loss = curve.last().unwrap().1;
        println!("Overfit final average loss: {final_loss:.6}");
        assert!(final_loss < 0.5, "five-sentence overfit target missed; loss curve: {curve:?}");
    }

    #[test]
    fn greedy_generation_reproduces_overfit_sentence_continuation() {
        const CORPUS: [&str; 5] = [
            "The trend is increasing",
            "The trend is decreasing",
            "The mean is stable",
            "An outlier changes the trend",
            "The trend remains significant",
        ];
        let tokenizer = Tokenizer::build(&CORPUS, 2048);
        let examples: Vec<_> = CORPUS.iter().map(|s| TrainingExample::from_text(&tokenizer, s)).collect();
        let mut model = Transformer::new(tokenizer.vocab_size(), 32, 2, 64, 1, 64, 42).unwrap();
        Trainer::train(&mut model, &examples, 200, 0.1, 0).unwrap();

        let (mut prompt, _) = tokenizer.tokenize("The mean is");
        prompt.pop(); // tokenize() appends EOS, but the prompt is a sentence prefix.
        let generated = model.generate(&prompt, 8).unwrap();
        let generated_text = tokenizer.detokenize(&generated);
        println!("Greedy generation from 'The mean is': {generated_text}");
        assert!(!generated.is_empty(), "generation must produce a continuation");
        assert!(generated.iter().all(|&id| id != tokenizer.unk_id() && id != tokenizer.pad_id()), "generation must not emit UNK or PAD: {generated:?}");
        assert_eq!(generated_text, "stable");
    }

    #[test]
    fn overfits_pipeline_generated_explanation_corpus_in_five_hundred_epochs() {
        let corpus_path = format!("{}/fixtures/corpus.txt", env!("CARGO_MANIFEST_DIR"));
        let corpus_text = std::fs::read_to_string(corpus_path).unwrap();
        let sentences: Vec<&str> = corpus_text.lines().collect();
        let tokenizer = Tokenizer::build(&sentences, 2048);
        let examples: Vec<_> = sentences
            .iter()
            .map(|sentence| TrainingExample::from_text(&tokenizer, sentence))
            .collect();
        let mut model = Transformer::new(tokenizer.vocab_size(), 32, 2, 64, 1, 64, 42).unwrap();
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

        for epoch in (100..=500).step_by(100) {
            Trainer::train(&mut model, &examples, 100, 0.1, 0).unwrap();
            println!("Real-corpus overfit loss epoch {epoch}: {:.6}", average_loss(&model));
        }

        let (mut prompt, _) = tokenizer.tokenize("The mean is");
        prompt.pop();
        let generated = model.generate(&prompt, 20).unwrap();
        println!(
            "Real-corpus greedy generation from 'The mean is': {}",
            tokenizer.detokenize(&generated)
        );

        let final_loss = average_loss(&model);
        println!("Real-corpus final average loss: {final_loss:.6}");
        assert!(final_loss < 0.5, "real-corpus overfit target missed: {final_loss}");
    }

    #[test]
    fn phase2_trains_on_real_corpus() {
        let corpus_path = format!("{}/fixtures/corpus_v2.txt", env!("CARGO_MANIFEST_DIR"));
        let corpus_text = std::fs::read_to_string(corpus_path).unwrap();
        let sentences: Vec<&str> = corpus_text.lines().collect();
        let tokenizer = Tokenizer::build(&sentences, 2048);
        let examples: Vec<_> = sentences
            .iter()
            .map(|sentence| TrainingExample::from_text(&tokenizer, sentence))
            .collect();
        let mut model = Transformer::new(tokenizer.vocab_size(), 128, 4, 512, 4, 64, 42).unwrap();
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

        for epoch in (100..=1000).step_by(100) {
            Trainer::train(&mut model, &examples, 100, 0.1, 0).unwrap();
            if epoch % 200 == 0 || epoch == 100 {
                println!("Phase 2 corpus loss epoch {epoch}: {:.6}", average_loss(&model));
            }
        }

        for prompt_text in ["The mean is", "The fit is"] {
            let (mut prompt, _) = tokenizer.tokenize(prompt_text);
            prompt.pop();
            let generated = model.generate(&prompt, 20).unwrap();
            println!("Phase 2 generation from '{prompt_text}': {}", tokenizer.detokenize(&generated));
        }

        let final_loss = average_loss(&model);
        println!("Phase 2 final average loss: {final_loss:.6}");
        assert!(final_loss < 0.5, "Phase 2 overfit target missed: {final_loss}");
    }

    #[test]
    #[ignore = "manual short Phase 2 matmul timing comparison"]
    fn phase2_matmul_five_epoch_timing() {
        use std::time::Instant;

        let corpus_path = format!("{}/fixtures/corpus_v2.txt", env!("CARGO_MANIFEST_DIR"));
        let corpus_text = std::fs::read_to_string(corpus_path).unwrap();
        let sentences: Vec<&str> = corpus_text.lines().collect();
        let tokenizer = Tokenizer::build(&sentences, 2048);
        let examples: Vec<_> = sentences
            .iter()
            .map(|sentence| TrainingExample::from_text(&tokenizer, sentence))
            .collect();

        for (name, legacy) in [("fast", false), ("legacy_scalar", true)] {
            crate::matmul::set_force_legacy_scalar(legacy);
            let mut model = Transformer::new(tokenizer.vocab_size(), 128, 4, 512, 4, 64, 42).unwrap();
            let start = Instant::now();
            Trainer::train(&mut model, &examples, 5, 0.1, 0).unwrap();
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "Phase 2 {name} matmul: {:.6} seconds total, {:.6} seconds per epoch",
                elapsed,
                elapsed / 5.0
            );
        }
        crate::matmul::set_force_legacy_scalar(false);
    }
}
