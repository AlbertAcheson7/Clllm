# Local state after resource separation

## Active paths

- Git repository: `/Users/joey/Documents/Clllm`
- Full datasets: `/Users/joey/Documents/Clllm_resources/shared_data`
- Shared model: `/Users/joey/Documents/Clllm_resources/shared_models/PubMedBERT`
- Test outputs: `/Users/joey/Documents/Clllm_resources/runs`
- Runtime checkpoints: `/Users/joey/Documents/Clllm_resources/archive`

The Git working directory is approximately 9 MB and contains no full dataset,
model weights, checkpoints, or generated run outputs.

## Verification

- Seven helper tests passed.
- NER and EL passed smoke tests using committed sample data.
- NER and EL passed smoke tests using the full local data paths.
- Both projects loaded the same external PubMedBERT model path.

Run the standard local checks with:

```bash
bash scripts/run_local_smoke.sh sample
bash scripts/run_local_smoke.sh real
```
