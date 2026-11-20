# TinyStories Stage 1

This path replaces the 36-word demo vocabulary with a vocabulary learned from
TinyStories training text. It trains a new checkpoint; the earlier
`grammar-stage1.chk` is left intact because its vocabulary and model shape are
different.

## Data

Run `scripts/download-corpora.ps1` from the repository root. It samples 2,000
training stories and 200 validation stories from evenly spaced offsets in the
official TinyStories splits, and samples 10 pairs from each of BLiMP's 67
paradigms. Downloads are kept under ignored `data/` so the multi-gigabyte source
corpus is not added to the project. TinyStories is synthetic text and is listed
under CDLA-Sharing-1.0; check its terms before redistributing it. BLiMP is
CC-BY and should be cited to Warstadt et al. (2020).

## Train and evaluate

```text
cargo run --release --bin tinystories-stage1 -- --stories 1000 --vocab-size 4096 --epochs 1
```

The model builds its tokenizer from the training stories only, uses a maximum
sequence length of 64, and saves both model weights and tokenizer vocabulary to
`target/tinystories-stage1.chk`. It reports validation next-token loss and the
BLiMP result. `--stories 2000` uses the full downloaded training sample;
`--vocab-size`, `--epochs`, and `--learning-rate` are configurable.

## Current scope and result

The first local run used 1,000 training stories, 200 validation stories, a
4,096-token vocabulary, and one epoch. Validation loss went from 8.3310 to
7.8935. Of 670 sampled BLiMP pairs, only 29 had no out-of-vocabulary words;
the model preferred the grammatical sentence on 17 of those 29 (58.6%). This is
too small a covered sample to make a reliable grammar claim. BLiMP remains
evaluation-only; no BLiMP sentences were used for training or vocabulary
construction.

The tokenizer is still word-level. A larger corpus will introduce more unique
words, so the cap maps rare ones to `<UNK>`. Improving BLiMP coverage will
require a subword tokenizer or a much broader training vocabulary. The full
TinyStories dataset contains about 2.1 million rows and 7.62 GB; this first run
uses a small sample because the from-scratch CPU model is slow at this scale.
