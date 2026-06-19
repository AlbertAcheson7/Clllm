# Server migration to `CLllm_v2`

The new directory is code-only. It reuses external data and one shared model.

## Completed deployment (2026-06-19)

The migration was completed and verified on `202.112.113.33`:

```text
~/CLllm_v2/                                      active code
~/shared_data/                                   large datasets
~/shared_models/PubMedBERT/                      shared model
~/runs/clllm_v2_{ner,el}_real_smoke/             active summaries
~/server_archive/legacy/CLllm_20260619/          old project
~/server_archive/artifacts/chroma_db_20260619/   old vector index
~/server_archive/runs/                           checkpoints and old runs
```

The server uses the `termalign` Conda environment. Both projects passed unit
tests and GPU smoke tests using the same shared PubMedBERT directory.

```text
~/CLllm_v2/                  active code
~/shared_data/               large NER and EL datasets
~/shared_models/PubMedBERT/  one shared model directory
~/runs/                      all generated results
~/server_archive/            old code and old runs
```

## Safe sequence

1. Inventory the old project:

   ```bash
   bash scripts/server_inventory.sh ~/CLllm
   ```

2. From the local machine, deploy to a new directory:

   ```bash
   bash scripts/deploy_clllm_v2.sh user@server CLllm_v2
   ```

3. On the server, run helper tests and the EL data audit:

   ```bash
   cd ~/CLllm_v2
   ~/anaconda3/bin/conda run -n termalign \
     python -m unittest discover -s projects -p 'test_*.py' -v
   ~/anaconda3/bin/conda run -n termalign \
     python projects/el_rag_contrastive/run.py \
     --dataset-dir ~/shared_data/ICD10-ICD11 \
     --output-dir ~/runs/el_audit \
     --validate-data-only
   ```

4. Run both smoke tests with the same model directory:

   ```bash
   ~/anaconda3/bin/conda run -n termalign \
     python projects/ner_span_pruner/run.py \
     --dict-train ~/shared_data/bc5cdr-DictMatching/train.txt \
     --chatgpt-train ~/shared_data/bc5cdr-ChatGPT/train.txt \
     --gold-dev ~/shared_data/BC5CDR/merged_bc5cdr_development_data.json \
     --gold-test ~/shared_data/BC5CDR/merged_bc5cdr_test_data.json \
     --model-name ~/shared_models/PubMedBERT \
     --output-dir ~/runs/ner_smoke \
     --smoke-test --local-files-only

   ~/anaconda3/bin/conda run -n termalign \
     python projects/el_rag_contrastive/run.py \
     --dataset-dir ~/shared_data/ICD10-ICD11 \
     --model-name ~/shared_models/PubMedBERT \
     --output-dir ~/runs/el_smoke \
     --smoke-test --local-files-only
   ```

5. Only after both pass, archive rather than delete the old code. This step was
   completed on 2026-06-19:

   ```bash
   mkdir -p ~/server_archive/legacy
   mv ~/CLllm ~/server_archive/legacy/CLllm_20260619
   ```

Do not move the old project if its `data/` or `model/` directories are still
the only copies. First move those resources into the shared directories and
rerun the smoke tests with the new absolute paths.
