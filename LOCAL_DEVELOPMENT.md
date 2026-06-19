# Local development

The local machine follows the same separation as the server:

```text
~/Documents/Clllm/                    Git repository
~/Documents/Clllm_resources/
├── shared_data/                      full local datasets
├── shared_models/PubMedBERT          shared model path
├── runs/                             local outputs
└── archive/                          local runtime archive
```

`archive/legacy/` inside the repository is different: it contains small,
deprecated source code that is intentionally versioned. Large runtime archives
belong in `Clllm_resources/archive/`.

## Tests

Helper tests do not load model weights:

```bash
conda run -n deepke-llm python -m unittest \
  projects.ner_span_pruner.tests.test_helpers \
  projects.el_rag_contrastive.tests.test_helpers -v
```

Run both projects with committed sample data:

```bash
bash scripts/run_local_smoke.sh sample
```

Run both projects against the full local datasets:

```bash
bash scripts/run_local_smoke.sh real
```

Override paths or environment when needed:

```bash
CLLLM_RESOURCES=/another/location \
CLLLM_CONDA_ENV=deepke-llm \
CLLLM_DEVICE=cpu \
bash scripts/run_local_smoke.sh real
```

After local tests pass, commit and push the code. The server then pulls the
same commit and uses its own `~/shared_data`, `~/shared_models`, and `~/runs`.
