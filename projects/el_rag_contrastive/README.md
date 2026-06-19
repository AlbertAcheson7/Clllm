# EL dense retrieval and contrastive reranking

## What this study implements

1. Read ICD10/ICD11 names, hierarchy paths, and known mappings.
2. Encode source and target concepts with PubMedBERT.
3. Retrieve a dense Top-K target candidate set ("RAG" in this experiment).
4. Train contrastively within that candidate space.
5. Rerank and evaluate Hits@K, Recall@20, and MRR.

This pipeline does not generate prose. An optional LLM filter exists, but the
core RAG stage is dense candidate retrieval.

## Validate the committed sample without a model

```bash
python projects/el_rag_contrastive/run.py \
  --output-dir /tmp/el_sample_audit \
  --validate-data-only
```

## Minimal model smoke test

```bash
python projects/el_rag_contrastive/run.py \
  --model-name /absolute/path/to/PubMedBERT \
  --output-dir /absolute/path/to/runs/el_smoke \
  --smoke-test --local-files-only
```

## Sample provenance

`sample_data/` is a deterministic slice of the local Integration2023 layout:
the first 10 supervision pairs, first 10 reference pairs, all involved source
and gold target entities, and enough deterministic target distractors to reach
40 targets. Rebuild it with:

```bash
python projects/el_rag_contrastive/build_sample_data.py \
  --dataset-dir /data/ICD10-ICD11
```

The slice is only for pipeline verification. Confirm upstream ICD and dataset
redistribution terms before publishing it outside the private repository.
