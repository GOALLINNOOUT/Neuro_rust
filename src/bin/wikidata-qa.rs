use std::io::{self, BufRead, Write};

const CAPITALS: &str = include_str!("../../fixtures/wikidata_capitals.tsv");

#[derive(Debug)]
struct Fact {
    country: String,
    country_qid: String,
    aliases: Vec<String>,
    capital: String,
    capital_qid: String,
    property: String,
}

fn normalize(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn load_facts() -> Result<Vec<Fact>, String> {
    CAPITALS
        .lines()
        .skip(1)
        .enumerate()
        .map(|(index, line)| {
            let fields: Vec<_> = line.split('\t').collect();
            if fields.len() != 6 {
                return Err(format!("invalid Wikidata TSV row {}", index + 2));
            }
            Ok(Fact {
                country: fields[0].to_string(),
                country_qid: fields[1].to_string(),
                aliases: fields[2]
                    .split('|')
                    .filter(|alias| !alias.is_empty())
                    .map(normalize)
                    .collect(),
                capital: fields[3].to_string(),
                capital_qid: fields[4].to_string(),
                property: fields[5].to_string(),
            })
        })
        .collect()
}

fn answer(question: &str, facts: &[Fact]) -> Option<String> {
    let normalized = normalize(question);
    let phrase = "capital of ";
    let offset = normalized.find(phrase)? + phrase.len();
    let country_query = normalized[offset..].trim();
    let fact = facts.iter().find(|fact| {
        normalize(&fact.country) == country_query
            || fact.aliases.iter().any(|alias| alias == country_query)
    })?;
    let sentence_end = if fact.capital.ends_with('.') { "" } else { "." };
    Some(format!(
        "The capital of {} is {}{}\nSource: https://www.wikidata.org/wiki/{}#{} ({} Q{} to {} Q{}).",
        fact.country,
        fact.capital,
        sentence_end,
        fact.country_qid,
        fact.property,
        fact.country,
        fact.country_qid.trim_start_matches('Q'),
        fact.capital,
        fact.capital_qid.trim_start_matches('Q')
    ))
}

fn respond(question: &str, facts: &[Fact]) {
    match answer(question, facts) {
        Some(response) => println!("{response}"),
        None => println!(
            "I don't have that fact in my local sample. For now, I can answer capital-of questions for 10 countries."
        ),
    }
}

fn run() -> Result<(), String> {
    let facts = load_facts()?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        respond(&args.join(" "), &facts);
        return Ok(());
    }

    println!(
        "Neuro_rust Wikidata QA (local sample: {} country capitals). Type 'exit' to quit.",
        facts.len()
    );
    println!("Try: What is the capital of France?");
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    write!(stdout, "> ").map_err(|error| format!("could not write prompt: {error}"))?;
    stdout
        .flush()
        .map_err(|error| format!("could not flush prompt: {error}"))?;
    for line in stdin.lock().lines() {
        let question = line.map_err(|error| format!("could not read question: {error}"))?;
        let question = question.trim();
        if question.eq_ignore_ascii_case("exit") || question.eq_ignore_ascii_case("quit") {
            break;
        }
        if question.is_empty() {
            continue;
        }
        respond(question, &facts);
        write!(stdout, "> ").map_err(|error| format!("could not write prompt: {error}"))?;
        stdout
            .flush()
            .map_err(|error| format!("could not flush prompt: {error}"))?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}
