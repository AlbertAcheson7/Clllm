# NER span pruner

## What this study implements

This is a span-level method, not the archived token-level BIO classifier.

1. Read noisy DictMatching and ChatGPT annotations.
2. Train PubMedBERT to score span/type compatibility.
3. Generate original, expanded, and shrunken boundary variants.
4. Select thresholds and NMS overlap on gold development data.
5. Freeze those choices and evaluate once on gold test data.

## Minimal smoke test

The default data paths point to the committed sample:

```bash
python projects/ner_span_pruner/run.py \
  --model-name /absolute/path/to/PubMedBERT \
  --output-dir /absolute/path/to/runs/ner_smoke \
  --smoke-test --local-files-only
```

For a full experiment, pass the four real-data paths explicitly. Data, model
weights, checkpoints, and outputs belong outside this repository.

## Sample provenance

`sample_data/` contains the first 10 document blocks from the local
BC5CDR-derived DictMatching and ChatGPT files, plus the first 10 rows of the
processed BC5CDR development and test JSON files. Rebuild it with:

```bash
python projects/ner_span_pruner/build_sample_data.py \
  --dict-train /data/bc5cdr-DictMatching/train.txt \
  --chatgpt-train /data/bc5cdr-ChatGPT/train.txt \
  --gold-dev /data/merged_bc5cdr_development_data.json \
  --gold-test /data/merged_bc5cdr_test_data.json
```

The slice is only for format and pipeline verification. Confirm upstream
redistribution terms before publishing it outside the private repository.
