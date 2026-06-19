# Current project layout

```text
projects/
├── ner_span_pruner/       active NER study
└── el_rag_contrastive/    active EL study

archive/legacy/            deprecated but recoverable code
scripts/                   data conversion and safe server operations
```

The active projects are independent. They may use different datasets while
both receiving the same external model directory through `--model-name`.

Large data, model weights, checkpoints, logs, and outputs are intentionally
outside Git. The only committed data is the deterministic test material under
each project's `sample_data/`.

Local runtime resources live under `~/Documents/Clllm_resources/`; server
runtime resources live under `~/shared_data`, `~/shared_models`, `~/runs`, and
`~/server_archive`. See `LOCAL_DEVELOPMENT.md` and `SERVER_MIGRATION.md`.
