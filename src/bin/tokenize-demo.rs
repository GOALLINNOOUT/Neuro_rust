use reason_engine::tokenizer::Tokenizer;

const CORPUS: [&str; 9] = [
    "The mean is 20.43",
    "The median is 18.00",
    "An increasing trend remains significant",
    "One outlier changes the slope",
    "The trend persists after excluding the outlier",
    "However the evidence is approximately stable",
    "The analysis confirms a rising pattern",
    "Mean and median summarize the values",
    "The trend is increasing at 20.43",
];

fn main() {
    let tokenizer = Tokenizer::build(&CORPUS, 2048);
    let example =
        "The increasing trend is significant at 20.43, however excluding -3.7 changes the mean";
    let (ids, numeric_values) = tokenizer.tokenize(example);

    println!("{}", tokenizer.build_summary());
    println!("Token IDs: {ids:?}");
    println!("Numeric values: {numeric_values:?}");
    println!("Detokenized: {}", tokenizer.detokenize(&ids));
}
