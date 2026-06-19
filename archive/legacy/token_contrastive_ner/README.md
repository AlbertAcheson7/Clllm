# Archived token-level contrastive NER baseline

- Archived: 2026-06-19
- Former location: `pre_experiment/`
- Files: `run_baseline.py`, `token_cl_model.py`, `mini_data.py`, `config.py`
- Status: deprecated research baseline
- Active replacement: `projects/ner_span_pruner/`

## Why it is different

This baseline aligns character annotations to token-level BIO labels, trains a
token classifier with cross-entropy, and adds supervised contrastive loss over
token representations. The active study instead creates and scores complete
span/type candidates, repairs boundaries through expansion/shrinkage, and
applies thresholding plus NMS.

It is archived for methodological comparison, not imported by active code.
To restore it, copy these four files back into one directory, install
`seqeval`, and supply the original BC5CDR JSON paths.
