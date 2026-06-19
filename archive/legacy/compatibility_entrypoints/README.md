# Archived compatibility entrypoints

- Archived: 2026-06-19
- Former location: `pre_experiment/run_span_pruner.py` and
  `pre_experiment/run_el_rag_contrastive.py`
- Reason: these files only forwarded execution into the old `experiments/`
  layout.
- Replacements:
  - `projects/ner_span_pruner/run.py`
  - `projects/el_rag_contrastive/run.py`

They are retained only to explain old server commands and must not be used as
active entrypoints.
