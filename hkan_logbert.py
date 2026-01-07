import argparse
import json
import hkan
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from tqdm import tqdm
from log import logger
from hkan_factory import build_hkan, get_linear_factory
from logbert_adapter import (
    build_split,
    concatenate_splits,
    determine_vocab_size,
    encode_split_sequences,
    generate_train_valid_sessions,
    load_vocab,
    load_windowed_split,
    make_masked_contexts,
    build_token_mapper,
)
from unsup_masked import HKANMaskedLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="HKAN adaptation aligned with LogBERT preprocessing and scoring.")
    parser.add_argument("--train-path", required=True, help="Path to the training sessions file (e.g., data/out/train).")
    parser.add_argument(
        "--test-normal-path", required=True, help="Path to the normal test sessions file (e.g., data/out/test_normal)."
    )
    parser.add_argument(
        "--test-abnormal-path",
        required=True,
        help="Path to the abnormal test sessions file (e.g., data/out/test_abnormal).",
    )
    parser.add_argument("--vocab-path", required=True, help="Path to vocab.pkl produced by LogBERT preprocessing.")
    parser.add_argument("--output-dir", default="runs_logbert", help="Directory to store run artifacts.")

    parser.add_argument("--window-size", type=int, default=20, help="Sliding window size for session slicing.")
    parser.add_argument("--seq-len", type=int, default=None, help="Maximum sequence length per session (after slicing).")
    parser.add_argument("--min-len", type=int, default=0, help="Minimum sequence length to keep a window.")

    parser.add_argument("--train-ratio", type=float, default=1.0, help="Fraction of training sessions to sample.")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Fraction of sampled sessions for validation.")

    parser.add_argument("--mask-ratio", type=float, default=0.65, help="Mask ratio for MLM training contexts.")
    parser.add_argument("--window-left", type=int, default=10, help="Number of tokens in the left context bag.")
    parser.add_argument("--window-right", type=int, default=10, help="Number of tokens in the right context bag.")
    parser.add_argument("--max-masks-per-sequence", type=int, help="Optional cap on masks per sequence during training.")
    parser.add_argument(
        "--eval-max-masks",
        type=int,
        help="Optional cap on masks per sequence during evaluation (defaults to no cap).",
    )

    parser.add_argument(
        "--num-candidates",
        type=int,
        default=5,
        help="Top-K candidates used to determine undetected tokens (LogBERT default = 30).",
    )
    parser.add_argument("--seq-threshold-min", type=float, default=0.0, help="Minimum sequence threshold to scan.")
    parser.add_argument("--seq-threshold-max", type=float, default=1.0, help="Maximum sequence threshold to scan.")
    parser.add_argument("--seq-threshold-step", type=float, default=0.05, help="Step size for threshold search grid.")
    parser.add_argument(
        "--seq-threshold-beta",
        type=float,
        default=2.0,
        help="Beta parameter for F-beta optimisation when selecting the best sequence threshold (use >1 to emphasise recall).",
    )

    parser.add_argument(
        "--disable-temperature-calibration",
        action="store_true",
        help="Disable HKAN MLM temperature calibration (median NLL).",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=["global", "chunk"],
        default="global",
        help="Temperature calibration strategy: single global scalar or per-chunk medians.",
    )
    parser.add_argument(
        "--calibration-chunk-size",
        type=int,
        default=100,
        help="Chunk size (in masked tokens) used for chunk-wise temperature calibration.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--linear-head",
        choices=["linear", "ridge", "elasticnet", "lars", "bayesridge"],
        default="linear",
        help="Linear model for the HKAN shared head.",
    )
    parser.add_argument(
        "--linear-alpha",
        type=float,
        default=1.0,
        help="Regularisation strength for ridge/elastic net heads.",
    )
    parser.add_argument(
        "--n-basis",
        type=int,
        default=32,
        help="Number of basis functions in ExpandingLayer.",
    )
    parser.add_argument(
        "--basis-fn",
        choices=["tanh", "gaussian", "sigmoid", "relu", "softplus", "identity"],
        default="tanh",
        help="Basis function used in ExpandingLayer.",
    )

    parser.add_argument(
        "--fixed-window",
        action="store_false",
        dest="adaptive_window",
        help="Disable adaptive windowing (defaults to adaptive).",
    )

    parser.add_argument(
        "--train_sample", 
        type=float, 
        default=1, 
        help="Whether to use a sample from the training set, with the percentage set here."
    )

    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=128,
        help="How many test data samples should each batch include."
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=2,
        help="How many threads to use when performing inference. Set to 1 for no parallelism"
    )
    parser.add_argument(
        "--use-gpu-expanding",
        action="store_true",
        help="Enable CuPy-accelerated expanding layer if available.",
    )
    parser.add_argument(
        "--gpu-chunk-size",
        type=int,
        default=None,
        help="Chunk size for GPU expanding layer to control memory use.",
    )

    parser.set_defaults(adaptive_window=True)

    args = parser.parse_args()
    if args.seq_threshold_step <= 0:
        parser.error("--seq-threshold-step must be positive.")
    if args.num_candidates <= 0:
        parser.error("--num-candidates must be positive.")
    return args


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_splits(args, rng: np.random.Generator):
    logger.info(f"Generating training and validation windows from {args.train_path}")
    train_logs, valid_logs, train_times, valid_times = generate_train_valid_sessions(
        args.train_path,
        window_size=args.window_size,
        adaptive_window=args.adaptive_window,
        sample_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seq_len=args.seq_len,
        min_len=args.min_len,
        rng=rng,
        train_sample=args.train_sample
    )

    train_split = build_split(train_logs, times=train_times)
    valid_split = build_split(valid_logs, times=valid_times)

    logger.info(f"Loading test windows from {args.test_normal_path} (normal) and {args.test_abnormal_path} (abnormal)")
    test_normal_split = load_windowed_split(
        args.test_normal_path,
        window_size=args.window_size,
        adaptive_window=args.adaptive_window,
        seq_len=args.seq_len,
        min_len=args.min_len,
        label=0,
    )
    test_abnormal_split = load_windowed_split(
        args.test_abnormal_path,
        window_size=args.window_size,
        adaptive_window=args.adaptive_window,
        seq_len=args.seq_len,
        min_len=args.min_len,
        label=1,
    )

    combined_test_split = concatenate_splits([test_normal_split, test_abnormal_split])

    return {
        "train": train_split,
        "valid": valid_split,
        "test_normal": test_normal_split,
        "test_abnormal": test_abnormal_split,
        "test_combined": combined_test_split,
    }


def build_contexts(
    split,
    vocab_size: int,
    *,
    rng: np.random.Generator,
    window_left: int,
    window_right: int,
    mask_probability: float,
    max_masks_per_sequence: Optional[int],
):
    return make_masked_contexts(
        split,
        vocab_size,
        window_left=window_left,
        window_right=window_right,
        mask_probability=mask_probability,
        max_masks_per_sequence=max_masks_per_sequence,
        rng=rng,
        include_quantitatives=False,
        include_semantics=False,
        include_parameters=False,
    )


def ensure_training_contexts(
    split,
    vocab_size: int,
    args,
    rng: np.random.Generator,
):
    X, y, seq_idx = build_contexts(
        split,
        vocab_size,
        rng=rng,
        window_left=args.window_left,
        window_right=args.window_right,
        mask_probability=args.mask_ratio,
        max_masks_per_sequence=args.max_masks_per_sequence,
    )
    if X.shape[0] == 0:
        logger.warning(
            f"No masked contexts sampled with mask_ratio={args.mask_ratio}; masking all positions for training."
        )
        X, y, seq_idx = build_contexts(
            split,
            vocab_size,
            rng=rng,
            window_left=args.window_left,
            window_right=args.window_right,
            mask_probability=args.mask_ratio,
            max_masks_per_sequence=args.max_masks_per_sequence,
        )
    if X.shape[0] == 0:
        raise RuntimeError("Masked language model has zero training contexts; check preprocessing options.")
    return X, y, seq_idx


def transform_with_scaler(scaler: RobustScaler, X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        return np.zeros((0, scaler.n_features_in_), dtype=np.float32)
    return scaler.transform(X)


def aggregate_sequence_stats(split, hits: np.ndarray, seq_index: np.ndarray) -> List[Dict[str, int]]:
    stats: List[Dict[str, int]] = []
    total_sequences = split.n_sequences
    for seq_id in tqdm(range(total_sequences)):
        mask = seq_index == seq_id
        masked_tokens = int(mask.sum())
        detected = int(hits[mask].sum()) if masked_tokens else 0
        undetected_tokens = int(masked_tokens - detected)
        length = len(split.sequentials[seq_id])
        label = int(split.labels[seq_id]) if split.labels is not None and len(split.labels) > seq_id else 0
        stats.append(
            {
                "sequence_index": seq_id,
                "masked_tokens": masked_tokens,
                "undetected_tokens": undetected_tokens,
                "total_logkey": length,
                "label": label,
            }
        )
    return stats


def compute_anomaly(stats: List[Dict[str, int]], seq_threshold: float) -> int:
    anomalies = 0
    for record in stats:
        masked = record["masked_tokens"]
        if masked == 0:
            continue
        if record["undetected_tokens"] > masked * seq_threshold:
            anomalies += 1
    return anomalies


def find_best_threshold(
    normal_stats: List[Dict[str, int]],
    abnormal_stats: List[Dict[str, int]],
    thresholds: np.ndarray,
    beta: float,
) -> Dict[str, float]:
    if thresholds.size == 0:
        raise ValueError("No thresholds provided for search.")
    if beta <= 0:
        raise ValueError("beta must be positive for F-beta.")
    beta_sq = beta * beta
    best = {
        "seq_threshold": float(thresholds[0]),
        "FP": 0,
        "TP": 0,
        "TN": len(normal_stats),
        "FN": len(abnormal_stats),
        "precision": 0.0,
        "recall": 0.0,
        "f_beta": 0.0,
    }
    best_score = -1.0

    for seq_th in thresholds:
        FP = compute_anomaly(normal_stats, seq_th)
        TP = compute_anomaly(abnormal_stats, seq_th)
        TN = len(normal_stats) - FP
        FN = len(abnormal_stats) - TP

        precision = TP / (TP + FP + 1e-9)
        recall = TP / (TP + FN + 1e-9)
        denom = beta_sq * precision + recall
        if denom == 0:
            f_beta = 0.0
        else:
            f_beta = (1 + beta_sq) * precision * recall / denom

        if f_beta > best_score:
            best_score = f_beta
            best = {
                "seq_threshold": float(seq_th),
                "FP": int(FP),
                "TP": int(TP),
                "TN": int(TN),
                "FN": int(FN),
                "precision": float(precision),
                "recall": float(recall),
                "f_beta": float(f_beta),
            }

    return best


def apply_threshold_to_stats(stats: List[Dict[str, int]], threshold: float) -> np.ndarray:
    preds = []
    for record in stats:
        masked = record["masked_tokens"]
        if masked == 0:
            preds.append(0)
            continue
        ratio = record["undetected_tokens"] / masked
        preds.append(1 if ratio > threshold else 0)
    return np.asarray(preds, dtype=np.int64)


def stats_to_dataframe(stats: List[Dict[str, int]], dataset: str) -> pd.DataFrame:
    df = pd.DataFrame(stats)
    if df.empty:
        return df
    df["dataset"] = dataset
    df["undetected_ratio"] = df["undetected_tokens"] / np.maximum(df["masked_tokens"], 1)
    return df


def unique_out_dir(base: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base.parent / f"{base.name}_{ts}"
    out.mkdir(parents=True, exist_ok=False)
    return out

def batched_token_in_topk(model, X, y, *, topk, mode, chunk_size=10_000):
    """
    Stream X/y through model.token_in_topk in manageable chunks to avoid
    storing the full logits/boolean mask in memory.
    """
    hits = []
    n = len(X)
    for chunk_idx, start in enumerate(tqdm(range(0, n, chunk_size))):
        stop = min(start + chunk_size, n)
        hits.append(
            model.token_in_topk(
                X[start:stop],
                y[start:stop],
                topk=topk,
                mode=mode,
                chunk_index=chunk_idx,
            )
        )
    return np.concatenate(hits, axis=0)

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    if args.calibration_chunk_size <= 0:
        raise ValueError("calibration_chunk_size must be positive.")
    if args.seq_threshold_beta <= 0:
        raise ValueError("seq_threshold_beta must be positive.")

    base_out = Path(args.output_dir)
    output_dir = unique_out_dir(base_out)

    # output_dir = ensure_dir(Path(args.output_dir))
    # (output_dir / "artifacts").mkdir(exist_ok=True)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump({"args": vars(args)}, fh, indent=2)

    splits = prepare_splits(args, rng)

    vocab = load_vocab(args.vocab_path)
    token_mapper = build_token_mapper(vocab)
    if token_mapper is not None:
        for split in splits.values():
            encode_split_sequences(split, token_mapper)
    all_sequences = (
        splits["train"].sequentials
        + splits["valid"].sequentials
        + splits["test_normal"].sequentials
        + splits["test_abnormal"].sequentials
    )
    vocab_size = determine_vocab_size(vocab, all_sequences)
    logger.info(f"Vocab size determined as {vocab_size}")

    # === Training contexts ===
    train_X_raw, train_y, train_seq_idx = ensure_training_contexts(splits["train"], vocab_size, args, rng)
    logger.info(f"Shape of training data {train_X_raw.shape}")

    scaler = RobustScaler(quantile_range=(5, 95))
    scaler.fit(train_X_raw)
    train_X = scaler.transform(train_X_raw)
    logger.info(
        f"Training data scaled."
    )

    # Map basis-fn name to HKAN basis instance with reasonable default scales
    basis_name = args.basis_fn.lower()
    if basis_name == "tanh":
        basis_inst = hkan.Tanh(s=50)
    elif basis_name == "gaussian":
        basis_inst = hkan.Gaussian(s=3)
    elif basis_name == "sigmoid":
        basis_inst = hkan.Sigmoid(s=1)
    elif basis_name == "relu":
        basis_inst = hkan.ReLU(s=1)
    elif basis_name == "softplus":
        basis_inst = hkan.Softplus(s=1)
    else:
        basis_inst = hkan.Identity()

    mlm_model = HKANMaskedLanguageModel(
        lambda: build_hkan(
            n_basis=int(args.n_basis),
            infer_batch_size=int(args.infer_batch_size),
            n_jobs=int(args.n_jobs),
            use_gpu=bool(args.use_gpu_expanding),
            gpu_chunk_size=args.gpu_chunk_size,
            basis_fn=basis_inst,
        ),
        vocab_size,
        linear_factory=get_linear_factory(args.linear_head, args.linear_alpha),
    )
    mlm_model.fit(
        train_X,
        train_y,
        calibrate_temperature=False,
    )

    try:            
        mlm_model.save(output_dir / "mlm_model.joblib")
    except Exception as ex:
        traceback.format_exc()
        logger.error(f"Model could not be saved. Reason: {ex}")

    logger.info(
        f"Trained HKAN MLM on {train_X.shape[0]} masked positions (temperature={round(mlm_model.temperature, 2)})"
    )

    # === Validation and test contexts ===
    def _eval_contexts(split_key: str):
        X_raw, y, seq_idx = build_contexts(
            splits[split_key],
            vocab_size,
            rng=rng,
            window_left=args.window_left,
            window_right=args.window_right,
            mask_probability=args.mask_ratio,
            max_masks_per_sequence=args.eval_max_masks,
        )
        X = transform_with_scaler(scaler, X_raw)
        return X, y, seq_idx

    valid_X, valid_y, valid_seq_idx = _eval_contexts("valid")

    inference_mode = "GPU" if args.use_gpu_expanding else "CPU"
    if args.disable_temperature_calibration:
        logger.info(
            f"Temperature calibration disabled; retaining temperature={round(mlm_model.temperature, 3)}"
        )
    else:
        mlm_model.calibrate_temperature(
            valid_X,
            valid_y,
            mode=args.calibration_mode,
            chunk_size=args.calibration_chunk_size,
            device_mode=inference_mode,
        )

    test_normal_X, test_normal_y, test_normal_seq_idx = _eval_contexts("test_normal")
    test_abnormal_X, test_abnormal_y, test_abnormal_seq_idx = _eval_contexts("test_abnormal")

    # === Token detection statistics ===
    logger.info(
        f"Predicting top-k across datasets."
    )
    
    # valid_hits = mlm_model.token_in_topk(valid_X, valid_y, topk=args.num_candidates, mode='GPU')    
    # test_normal_hits = mlm_model.token_in_topk(test_normal_X, test_normal_y, topk=args.num_candidates)
    # test_abnormal_hits = mlm_model.token_in_topk(test_abnormal_X, test_abnormal_y, topk=args.num_candidates)

    valid_hits = batched_token_in_topk(
        mlm_model,
        valid_X,
        valid_y,
        topk=args.num_candidates,
        mode=inference_mode,
    )
    test_normal_hits = batched_token_in_topk(
        mlm_model,
        test_normal_X,
        test_normal_y,
        topk=args.num_candidates,
        mode=inference_mode,
    )
    test_abnormal_hits = batched_token_in_topk(
        mlm_model,
        test_abnormal_X,
        test_abnormal_y,
        topk=args.num_candidates,
        mode=inference_mode,
    )

    logger.info(
        f"Aggregate sequence statistics across datasets."
    )

    logger.info(f"validation set:")
    valid_stats = aggregate_sequence_stats(splits["valid"], valid_hits, valid_seq_idx)
    logger.info(f"Test (normal) set:")
    normal_stats = aggregate_sequence_stats(splits["test_normal"], test_normal_hits, test_normal_seq_idx)
    logger.info(f"Test (abnormal) set:")
    abnormal_stats = aggregate_sequence_stats(splits["test_abnormal"], test_abnormal_hits, test_abnormal_seq_idx)

    thresholds = np.arange(
        args.seq_threshold_min,
        args.seq_threshold_max + 1e-9,
        args.seq_threshold_step,
    )
    if thresholds.size == 0:
        thresholds = np.asarray([args.seq_threshold_min], dtype=float)

    logger.info(
        f"Searching for the best anomaly threshold."
    )
    best = find_best_threshold(normal_stats, abnormal_stats, thresholds, beta=args.seq_threshold_beta)
    best_threshold = best["seq_threshold"]
    logger.info(
        f'Best sequence threshold {round(best_threshold, 3)} -> Precision {round(best["precision"], 2)} | Recall {round(best["recall"], 2)} | F{args.seq_threshold_beta} {round(best["f_beta"], 2)}',
    )

    valid_fp = compute_anomaly(valid_stats, best_threshold)
    logger.info(f"Validation false positives at threshold: {valid_fp} / {len(valid_stats)}")

    test_stats = normal_stats + abnormal_stats
    test_labels = np.concatenate(
        [np.zeros(len(normal_stats), dtype=np.int64), np.ones(len(abnormal_stats), dtype=np.int64)]
    )
    test_preds = apply_threshold_to_stats(test_stats, best_threshold)

    TP = int(((test_preds == 1) & (test_labels == 1)).sum())
    FP = int(((test_preds == 1) & (test_labels == 0)).sum())
    TN = int(((test_preds == 0) & (test_labels == 0)).sum())
    FN = int(((test_preds == 0) & (test_labels == 1)).sum())
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )

    test_datasets = ["test_normal"] * len(normal_stats) + ["test_abnormal"] * len(abnormal_stats)
    test_df = stats_to_dataframe(test_stats, dataset="mixed")
    if not test_df.empty:
        test_df["dataset"] = test_datasets
        test_df["label"] = test_labels
        test_df["prediction"] = test_preds
    valid_df = stats_to_dataframe(valid_stats, dataset="valid")

    test_df.to_csv(output_dir / "test_sequence_stats.csv", index=False)
    valid_df.to_csv(output_dir / "valid_sequence_stats.csv", index=False)

    summary = {
        "vocab_size": int(vocab_size),
        "train_sequences": splits["train"].n_sequences,
        "valid_sequences": splits["valid"].n_sequences,
        "test_normal_sequences": splits["test_normal"].n_sequences,
        "test_abnormal_sequences": splits["test_abnormal"].n_sequences,
        "train_contexts": int(train_X.shape[0]),
        "valid_contexts": int(valid_X.shape[0]),
        "test_normal_contexts": int(test_normal_X.shape[0]),
        "test_abnormal_contexts": int(test_abnormal_X.shape[0]),
        "mask_ratio": args.mask_ratio,
        "window_size": args.window_size,
        "adaptive_window": args.adaptive_window,
        "num_candidates": args.num_candidates,
        "best_seq_threshold": best_threshold,
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": TP,
            "fp": FP,
            "tn": TN,
            "fn": FN,
        },
        "valid_false_positives": int(valid_fp),
        "mlm_temperature": float(mlm_model.temperature),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info(
        f"Final metrics -> Precision {round(precision, 2)} | Recall {round(recall, 2)} | F1 {round(f1, 2)} (TP={round(TP, 2)}, FP={round(FP, 2)}, TN={round(TN, 2)}, FN={round(FN, 2)})"
    )
    logger.info(f"Artifacts written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()


