# Server state after migration

Host: `202.112.113.33`
Environment: `termalign`

## Active

- Code: `~/CLllm_v2` (approximately 488 KB)
- Data: `~/shared_data` (approximately 1.3 GB)
- Shared model: `~/shared_models/PubMedBERT`
- Result summaries: `~/runs/clllm_v2_*`

## Archived, not deleted

- Old project: `~/server_archive/legacy/CLllm_20260619`
- Original inventory: `~/server_archive/inventory/CLllm_20260619`
- Chroma index: `~/server_archive/artifacts/chroma_db_20260619`
- Old pre-experiment runs: `~/server_archive/runs/20260618_preexperiments`
- Smoke-test checkpoints:
  `~/server_archive/runs/clllm_v2_smoke_checkpoints_20260619`

## Verification

- Seven project unit tests passed on the server.
- Both projects passed GPU smoke tests with committed sample data.
- Both projects passed GPU smoke tests with real server data.
- Both projects loaded the same model directory:
  `~/shared_models/PubMedBERT`.

Small-sample metrics verify execution only and must not be reported as final
research results.
