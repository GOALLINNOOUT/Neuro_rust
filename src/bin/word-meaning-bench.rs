use reason_engine::tokenizer::Tokenizer;
use reason_engine::trainer::{Trainer, TrainingExample};
use reason_engine::transformer::Transformer;

const TRAINING_TEXT: [&str; 30] = [
    "Brisk means quick and energetic",
    "Ancient means very old",
    "Fragile means easily broken",
    "Vast means very large",
    "Calm means peaceful and relaxed",
    "Scarce means hard to find",
    "Drowsy means sleepy",
    "Generous means willing to give",
    "Narrow means not wide",
    "Gloomy means dark and sad",
    "Rapid means very fast",
    "Tiny means very small",
    "Cautious means careful",
    "Delicate means easily damaged",
    "Silent means making no sound",
    "Beneath means under something",
    "Purchase means to buy",
    "Repair means to fix",
    "Observe means to watch",
    "Predict means to say what may happen",
    "Brisk is similar in meaning to rapid",
    "Vast is similar in meaning to huge",
    "Fragile is similar in meaning to delicate",
    "Purchase is similar in meaning to buy",
    "Repair is similar in meaning to fix",
    "Observe is similar in meaning to watch",
    "Ancient is the opposite of modern",
    "Generous is the opposite of selfish",
    "Scarce is the opposite of common",
    "Calm is the opposite of anxious",
];

#[derive(Clone, Copy)]
struct Question {
    task: &'static str,
    prompt: &'static str,
    choices: [&'static str; 4],
    correct: usize,
}

const QUESTIONS: [Question; 24] = [
    Question {
        task: "definition",
        prompt: "quick and energetic means",
        choices: ["brisk", "ancient", "fragile", "vast"],
        correct: 0,
    },
    Question {
        task: "definition",
        prompt: "very old means",
        choices: ["calm", "ancient", "narrow", "rapid"],
        correct: 1,
    },
    Question {
        task: "definition",
        prompt: "easily broken means",
        choices: ["gloomy", "tiny", "fragile", "silent"],
        correct: 2,
    },
    Question {
        task: "definition",
        prompt: "very large means",
        choices: ["scarce", "drowsy", "cautious", "vast"],
        correct: 3,
    },
    Question {
        task: "definition",
        prompt: "peaceful and relaxed means",
        choices: ["calm", "brisk", "delicate", "beneath"],
        correct: 0,
    },
    Question {
        task: "definition",
        prompt: "hard to find means",
        choices: ["generous", "scarce", "rapid", "narrow"],
        correct: 1,
    },
    Question {
        task: "definition",
        prompt: "easily damaged means",
        choices: ["vast", "silent", "delicate", "ancient"],
        correct: 2,
    },
    Question {
        task: "definition",
        prompt: "under something means",
        choices: ["brisk", "vast", "tiny", "beneath"],
        correct: 3,
    },
    Question {
        task: "synonym",
        prompt: "similar in meaning to brisk",
        choices: ["rapid", "huge", "delicate", "buy"],
        correct: 0,
    },
    Question {
        task: "synonym",
        prompt: "similar in meaning to vast",
        choices: ["fix", "huge", "watch", "old"],
        correct: 1,
    },
    Question {
        task: "synonym",
        prompt: "similar in meaning to fragile",
        choices: ["modern", "common", "delicate", "sleepy"],
        correct: 2,
    },
    Question {
        task: "synonym",
        prompt: "similar in meaning to purchase",
        choices: ["rapid", "huge", "fix", "buy"],
        correct: 3,
    },
    Question {
        task: "synonym",
        prompt: "similar in meaning to repair",
        choices: ["fix", "watch", "give", "old"],
        correct: 0,
    },
    Question {
        task: "synonym",
        prompt: "similar in meaning to observe",
        choices: ["buy", "watch", "predict", "find"],
        correct: 1,
    },
    Question {
        task: "synonym",
        prompt: "similar in meaning to generous",
        choices: ["brisk", "tiny", "willing", "modern"],
        correct: 2,
    },
    Question {
        task: "synonym",
        prompt: "similar in meaning to drowsy",
        choices: ["rapid", "huge", "common", "sleepy"],
        correct: 3,
    },
    Question {
        task: "antonym",
        prompt: "opposite of ancient",
        choices: ["modern", "old", "fragile", "gloomy"],
        correct: 0,
    },
    Question {
        task: "antonym",
        prompt: "opposite of generous",
        choices: ["careful", "selfish", "peaceful", "energetic"],
        correct: 1,
    },
    Question {
        task: "antonym",
        prompt: "opposite of scarce",
        choices: ["scarce", "tiny", "common", "silent"],
        correct: 2,
    },
    Question {
        task: "antonym",
        prompt: "opposite of calm",
        choices: ["relaxed", "sleepy", "dark", "anxious"],
        correct: 3,
    },
    Question {
        task: "antonym",
        prompt: "opposite of narrow",
        choices: ["wide", "small", "rapid", "careful"],
        correct: 0,
    },
    Question {
        task: "antonym",
        prompt: "opposite of gloomy",
        choices: ["sad", "calm", "dark", "old"],
        correct: 1,
    },
    Question {
        task: "antonym",
        prompt: "opposite of tiny",
        choices: ["rapid", "calm", "vast", "fragile"],
        correct: 2,
    },
    Question {
        task: "antonym",
        prompt: "opposite of vast",
        choices: ["gloomy", "careful", "ancient", "tiny"],
        correct: 3,
    },
];

fn content_ids(tokenizer: &Tokenizer, text: &str) -> Vec<usize> {
    let (ids, _) = tokenizer.tokenize(text);
    ids[1..ids.len() - 1].to_vec()
}

fn candidate_score(model: &Transformer, tokenizer: &Tokenizer, prompt: &str, choice: &str) -> f64 {
    let prompt_ids = content_ids(tokenizer, prompt);
    let choice_ids = content_ids(tokenizer, choice);
    if prompt_ids.contains(&tokenizer.unk_id())
        || choice_ids.is_empty()
        || choice_ids.contains(&tokenizer.unk_id())
    {
        return f64::NEG_INFINITY;
    }

    let mut ids = Vec::with_capacity(prompt_ids.len() + choice_ids.len() + 2);
    ids.push(tokenizer.bos_id());
    ids.extend(prompt_ids.iter().copied());
    ids.extend(choice_ids.iter().copied());
    ids.push(tokenizer.eos_id());
    let numbers = vec![None; ids.len()];
    let Ok(logits) = model.forward(&ids, &numbers) else {
        return f64::NEG_INFINITY;
    };

    // Position prompt_ids.len() predicts the first choice token after BOS + prompt.
    let first_choice = prompt_ids.len() + 1;
    let log_probability: f64 = choice_ids
        .iter()
        .enumerate()
        .map(|(offset, &target)| {
            let row = &logits[first_choice + offset - 1];
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_sum_exp = max
                + row
                    .iter()
                    .map(|value| (value - max).exp())
                    .sum::<f64>()
                    .ln();
            row[target] - log_sum_exp
        })
        .sum();
    log_probability / choice_ids.len() as f64
}

fn main() -> Result<(), String> {
    const SEED: u64 = 42;
    const EPOCHS: usize = 400;
    const LEARNING_RATE: f64 = 0.01;

    let tokenizer = Tokenizer::build(&TRAINING_TEXT, 512);
    let mut answer_position_counts = [0usize; 4];
    for question in QUESTIONS {
        answer_position_counts[question.correct] += 1;
        if content_ids(&tokenizer, question.prompt).contains(&tokenizer.unk_id())
            || question
                .choices
                .iter()
                .any(|choice| content_ids(&tokenizer, choice).contains(&tokenizer.unk_id()))
        {
            let (ids, _) = tokenizer.tokenize(question.prompt);
            return Err(format!(
                "question has words missing from the training vocabulary: {} ({ids:?}; {})",
                question.prompt,
                tokenizer.build_summary()
            ));
        }
    }
    if answer_position_counts
        .iter()
        .any(|count| *count != QUESTIONS.len() / 4)
    {
        return Err(format!(
            "correct answer positions are not balanced: {answer_position_counts:?}"
        ));
    }
    let examples: Vec<_> = TRAINING_TEXT
        .iter()
        .map(|text| TrainingExample::from_text(&tokenizer, text))
        .collect();
    let mut model = Transformer::new(tokenizer.vocab_size(), 32, 2, 64, 1, 32, SEED)?;
    Trainer::train(&mut model, &examples, EPOCHS, LEARNING_RATE, 0)?;

    println!("Word Meaning Benchmark v1");
    println!("Training examples: {}; evaluation questions: {}; vocab: {}; seed: {}; epochs: {}; learning rate: {}", TRAINING_TEXT.len(), QUESTIONS.len(), tokenizer.vocab_size(), SEED, EPOCHS, LEARNING_RATE);
    println!("Random-choice reference: 25.0% (four options per question)");

    let mut correct_by_task = [0usize; 3];
    let mut count_by_task = [0usize; 3];
    for (index, question) in QUESTIONS.iter().enumerate() {
        let scores: Vec<f64> = question
            .choices
            .iter()
            .map(|choice| candidate_score(&model, &tokenizer, question.prompt, choice))
            .collect();
        let predicted = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let task_index = match question.task {
            "definition" => 0,
            "synonym" => 1,
            _ => 2,
        };
        count_by_task[task_index] += 1;
        if predicted == question.correct {
            correct_by_task[task_index] += 1;
        }
        println!(
            "Q{:02} [{}] predicted={} expected={} scores={:?} | {}",
            index + 1,
            question.task,
            question.choices[predicted],
            question.choices[question.correct],
            scores
                .iter()
                .map(|score| format!("{score:.3}"))
                .collect::<Vec<_>>(),
            question.prompt
        );
    }

    let total_correct: usize = correct_by_task.iter().sum();
    println!(
        "Accuracy: {total_correct}/{} ({:.1}%)",
        QUESTIONS.len(),
        100.0 * total_correct as f64 / QUESTIONS.len() as f64
    );
    for (index, task) in ["definition", "synonym", "antonym"].iter().enumerate() {
        println!(
            "{task} accuracy: {}/{} ({:.1}%)",
            correct_by_task[index],
            count_by_task[index],
            100.0 * correct_by_task[index] as f64 / count_by_task[index] as f64
        );
    }
    Ok(())
}
