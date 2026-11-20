# Grammar Stage 1

This phase gives the from-scratch model short English sentences, then measures
whether it prefers grammatical forms it did not see verbatim during training.
The train set covers basic intransitive agreement, adjective-before-noun order,
and subject-verb-object order. The original held-out set has 24 basic minimal
pairs. A separate compositional set adds 48 harder pairs.

The score is the mean next-token negative log-likelihood of a sentence. A pair
counts correct when the grammatical sentence has lower loss. Random choice is
50%. The compositional split in `fixtures/grammar_stage1_compositional.tsv`
holds out subject/verb/object combinations while keeping every word in the
training vocabulary. It tests agreement and word order on combinations absent
from the original training sentences. The guidance examples are a separate set
of correct combinations and do not overlap those held-out combinations.

Accuracy alone can saturate on a small benchmark, so the runner also reports
the mean loss margin (incorrect sentence loss minus correct sentence loss) for
each category. Larger positive margins mean stronger preference. This remains
a small controlled diagnostic, not a claim of general language understanding;
the word-level tokenizer and one-layer model limit what can be learned from it.

Run from the repository root:

```text
cargo run --release --bin grammar-stage1
```

The default run trains from scratch, saves `target/grammar-stage1.chk`, reloads
that checkpoint, checks that its output matches before saving, and reports
held-out accuracy. Continue training from a saved checkpoint with:

```text
cargo run --release --bin grammar-stage1 -- --resume target/grammar-stage1.chk
```

Use `--epochs N`, `--learning-rate RATE`, or `--checkpoint PATH` to change the
training settings or checkpoint location. The original training examples and
held-out pairs are in `fixtures/grammar_stage1_train.txt` and
`fixtures/grammar_stage1_eval.tsv`.

To evaluate a saved model on the compositional split without updating it:

```text
cargo run --release --bin grammar-stage1 -- --resume target/grammar-stage1.chk --eval fixtures/grammar_stage1_compositional.tsv --eval-only
```

To continue training on the separate guidance combinations and then rescore:

```text
cargo run --release --bin grammar-stage1 -- --resume target/grammar-stage1.chk --guidance fixtures/grammar_stage1_guidance.txt --eval fixtures/grammar_stage1_compositional.tsv --epochs 100 --learning-rate 0.001 --checkpoint target/grammar-stage1-guided.chk
```

The CLI keeps the checkpoint vocabulary fixed when resuming. The benchmark
scores the held-out combinations before and after guidance; retain the earlier
checkpoint if the guided run lowers the measured held-out margins.
