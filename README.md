# HKAN-LogAD

HKAN-LogAD is an adaptation of the HKAN model to LogBERT-style preprocessing and scoring. The main entry point is `hkan_logbert.py`, which trains a masked-language-model style HKAN and evaluates anomaly detection using LogBERT-inspired sequence thresholds.

## Repository layout

- `hkan_logbert.py`: primary training/evaluation script aligned with LogBERT preprocessing.
- `logbert_adapter.py`: helpers for loading LogBERT-style data and vocabularies.
- `hkan.py`, `hkan_factory.py`: HKAN model and factory utilities.
- `unsup_masked.py`: HKAN masked language model wrapper.
- `requirements.txt`: Python dependencies.
- `runs/`: example run artifacts.

## Requirements

- Python 3.9+ recommended.
- Dependencies from `requirements.txt`.
- Optional GPU: `cupy` is listed but requires a CUDA-compatible environment. If you do not have CUDA, you can remove `cupy` from the requirements or install a CPU-only environment and avoid `--use-gpu-expanding`.

## Install

```bash
python -m venv .venv
. .venv/bin/activate  # On Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data format

Please visit https://github.com/HelenGuohx/logbert/tree/main for more details on where to download and how the HDFS dataset is structured.

## Run

Example (CPU):

```bash
python hkan_logbert.py \
  --train-path data/out/train \
  --test-normal-path data/out/test_normal \
  --test-abnormal-path data/out/test_abnormal \
  --vocab-path data/out/vocab.pkl \
  --output-dir runs_logbert
```

Example (GPU expanding layer):

```bash
python hkan_logbert.py \
  --train-path data/out/train \
  --test-normal-path data/out/test_normal \
  --test-abnormal-path data/out/test_abnormal \
  --vocab-path data/out/vocab.pkl \
  --output-dir runs_logbert \
  --use-gpu-expanding
```

## Outputs

Each run writes to a timestamped directory: `runs_logbert_YYYYMMDD_HHMMSS/`.

Artifacts include:

- `run_config.json`: the full CLI arguments.
- `mlm_model.joblib`: trained HKAN MLM (if serialization succeeds).
- `valid_sequence_stats.csv`: per-sequence stats for validation data.
- `test_sequence_stats.csv`: per-sequence stats for test data.
- `summary.json`: aggregate metrics and the chosen anomaly threshold.

## Useful flags

- `--window-size`, `--seq-len`, `--min-len`: control session slicing.
- `--mask-ratio`: fraction of tokens masked for MLM training.
- `--num-candidates`: top-k tokens for anomaly detection.
- `--seq-threshold-*`: search range for sequence anomaly threshold.
- `--linear-head`, `--linear-alpha`, `--n-basis`, `--basis-fn`: HKAN head configuration.
- `--infer_batch_size`, `--n_jobs`: inference batching and parallelism.

## Notes

- If you see zero masked contexts, lower `--mask-ratio` or ensure sessions are long enough.
- For CPU-only environments, omit `--use-gpu-expanding` and consider removing `cupy` from the dependency list.
