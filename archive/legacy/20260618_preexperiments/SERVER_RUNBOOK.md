# Server runbook

This directory is an isolated deployment unit. Upload `experiments/` only.
Do not overwrite the server's existing `data/`, `model/`, `src/`, or old
experiment outputs.

## 1. Safe upload

Upload into a new timestamped directory rather than the old project tree:

```bash
SERVER_HOST=user@server
SERVER_ROOT=/home/user/Clllm_runs/20260618_preexperiments

ssh "$SERVER_HOST" "mkdir -p '$SERVER_ROOT'"
rsync -av --delete \
  --exclude '__pycache__/' \
  experiments/ "$SERVER_HOST:$SERVER_ROOT/experiments/"
```

`--delete` is scoped only to the newly created remote `experiments/` directory.
It never touches server datasets or legacy code.

Record the local revision and working-tree diff before upload:

```bash
git rev-parse HEAD
git status --short
```

## 2. Server environment check

```bash
cd /home/user/Clllm_runs/20260618_preexperiments
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); print(torch.cuda.is_available(), torch.cuda.device_count())"
python -m unittest experiments.test_pre_experiments -v
```

Use the server's existing environment if it already contains PyTorch and
Transformers. Do not reinstall packages during a running experiment unless an
import actually fails.

## 3. NER

The NER task needs four paths:

- DictMatching train CoNLL
- ChatGPT train CoNLL
- BC5CDR gold development JSON
- BC5CDR gold test JSON

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/ner/run_span_pruner.py \
  --dict-train /absolute/server/path/bc5cdr-DictMatching/train.txt \
  --chatgpt-train /absolute/server/path/bc5cdr-ChatGPT/train.txt \
  --gold-dev /absolute/server/path/merged_bc5cdr_development_data.json \
  --gold-test /absolute/server/path/merged_bc5cdr_test_data.json \
  --model-name /absolute/server/path/PubMedBERT \
  --output-dir /absolute/server/path/runs/ner_smoke \
  --smoke-test --local-files-only
```

Full run:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python experiments/ner/run_span_pruner.py \
  --dict-train /absolute/server/path/bc5cdr-DictMatching/train.txt \
  --chatgpt-train /absolute/server/path/bc5cdr-ChatGPT/train.txt \
  --gold-dev /absolute/server/path/merged_bc5cdr_development_data.json \
  --gold-test /absolute/server/path/merged_bc5cdr_test_data.json \
  --model-name /absolute/server/path/PubMedBERT \
  --epochs 5 --batch-size 8 \
  --output-dir /absolute/server/path/runs/ner_full \
  > /absolute/server/path/runs/ner_full.log 2>&1 &
```

## 4. EL

Keep the downloaded ICD10-ICD11 dataset outside this code directory.

First validate its files:

```bash
python experiments/el/run_rag_contrastive.py \
  --dataset-dir /absolute/server/path/ICD10-ICD11 \
  --output-dir /absolute/server/path/runs/el_audit \
  --validate-data-only
```

The automatic layout recognizes:

```text
ent_ids_1.txt
ent_ids_2.txt
paths_1.txt
paths_2.txt
sup_pairs.txt
ref_pairs.txt
```

The local Integration2023 data contains usable names in `ent_ids_*.txt` and
hierarchy context in `paths_*.txt`. The default `--text-mode name_path` uses
both. Run `--text-mode name` as a name-only ablation if time permits.

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/el/run_rag_contrastive.py \
  --dataset-dir /absolute/server/path/ICD10-ICD11 \
  --model-name /absolute/server/path/PubMedBERT \
  --output-dir /absolute/server/path/runs/el_smoke \
  --smoke-test --local-files-only
```

Full run:

```bash
CUDA_VISIBLE_DEVICES=1 nohup python experiments/el/run_rag_contrastive.py \
  --dataset-dir /absolute/server/path/ICD10-ICD11 \
  --model-name /absolute/server/path/PubMedBERT \
  --text-mode name_path --epochs 3 --top-k 20 \
  --output-dir /absolute/server/path/runs/el_full \
  > /absolute/server/path/runs/el_full.log 2>&1 &
```

## 5. Result files

```text
runs/ner_full/results.csv
runs/ner_full/metrics.json
runs/el_full/results.csv
runs/el_full/metrics.json
```

Never copy results back into a data-source directory. Keep `code`, `data`,
`models`, and `runs` as separate server directories.
