# Word Meaning Benchmark v1

This benchmark checks whether the current from-scratch, word-level Transformer
can use short English training examples to select a word with the requested
meaning or relation. It is a small diagnostic benchmark, not a broad measure of
language understanding.

## Tasks

- **Definition reversal:** choose the word that matches a short definition.
- **Synonym:** select a word with a similar meaning.
- **Antonym:** select a word with the opposite meaning.

Training examples and evaluation questions are fixed in
`src/bin/word-meaning-bench.rs`. Evaluation uses different prompt wording from
training. The four answer positions are balanced, and each question has four
choices. The score is the model's mean next-token log probability for the
choice, normalized by the number of choice tokens. No answers are added to the
training text as question-answer pairs.

## Protocol

- No external data or dependencies; examples are hand-written and auditable.
- The vocabulary is built from training text only.
- The model is initialized with a fixed seed and trained from scratch for the
  fixed epoch count in the runner.
- Report accuracy overall and separately for each task. Uniform chance is 25%.
- Treat this as a reproducible baseline, not proof of general English ability:
  the lexicon is small, the data is synthetic, and training/testing use the
  same concepts with different prompt wording.

Run with:

```text
cargo run --release --bin word-meaning-bench
```

The runner prints the seed, training setup, per-task counts, and exact
predictions so results can be inspected rather than reduced to one score.
