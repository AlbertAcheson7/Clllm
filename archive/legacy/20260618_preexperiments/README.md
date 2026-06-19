# Isolated pre-experiments

```text
experiments/
├── ner/
│   └── run_span_pruner.py
├── el/
│   └── run_rag_contrastive.py
├── test_pre_experiments.py
├── requirements.txt
└── SERVER_RUNBOOK.md
```

The NER and EL tasks do not import the repository's legacy `src/` code and do
not assume that data lives beside the scripts. All server data, model, and
output locations can be supplied as absolute command-line paths.

The EL loader directly supports the local Integration2023 layout:

```text
ent_ids_1.txt
ent_ids_2.txt
paths_1.txt
paths_2.txt
sup_pairs.txt
ref_pairs.txt
```

By default it encodes the term name plus its hierarchy path. Use
`--text-mode name` for a name-only ablation.

See `SERVER_RUNBOOK.md` for the safe upload and execution procedure.
