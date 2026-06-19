# Independent research projects

This directory contains the two active research projects. They are deliberately
independent: neither project imports code from the other, from `src/`, or from
the archived token-level baseline.

- `ner_span_pruner`: span-level candidate generation, boundary correction,
  scoring, thresholding, and NMS for noisy BC5CDR NER annotations.
- `el_rag_contrastive`: ICD10-to-ICD11 dense retrieval followed by contrastive
  reranking of the retrieved Top-K candidates.

Both commands accept `--model-name`. On the server, pass the same external
PubMedBERT directory to both commands rather than copying model weights into
either project.

The committed `sample_data/` directories are deterministic minimal slices for
testing only. Their metrics must not be used as research results.
