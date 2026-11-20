# Wikidata QA starter

This is the first small, auditable knowledge-retrieval slice for `Neuro_rust`.
It contains ten country-to-capital facts exported from Wikidata's `capital`
property (P36) and queried locally. The TSV stores both entity IDs so each
answer can point back to its Wikidata source item.

This slice does **not** train the Transformer. The current model is too small
and uses a word-level tokenizer; memorizing these rows would be a less reliable
way to answer exact factual questions than looking them up. The CLI currently
supports capital-of questions and reports when a fact is outside this local
sample. Conversational wording and other question types can be added after
this lookup path is established.

## Run

Ask one question:

```text
cargo run --release --bin wikidata-qa -- "What is the capital of France?"
```

Start an interactive prompt:

```text
cargo run --release --bin wikidata-qa
```

Type `exit` or `quit` to finish.

## Provenance and license

The facts were retrieved from the official Wikidata Query Service on
2026-11-06. The extraction selected capital claims for these country items:
Q16, Q17, Q29, Q30, Q38, Q142, Q145, Q183, Q408, and Q668. The structured data
in Wikidata is available under CC0. The runner uses the local snapshot so its
answers are deterministic and do not require a network connection.

Reproduction query:

```sparql
SELECT ?country ?countryLabel ?capital ?capitalLabel WHERE {
  VALUES ?country {
    wd:Q30 wd:Q145 wd:Q142 wd:Q183 wd:Q17
    wd:Q16 wd:Q29 wd:Q38 wd:Q408 wd:Q668
  }
  ?country wdt:P36 ?capital .
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en".
  }
}
```
