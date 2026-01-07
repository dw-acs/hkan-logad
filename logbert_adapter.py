from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split

from bert_pytorch.dataset import WordVocab


@dataclass
class LogBERTSplit:
    """
    Container mirroring the structure emitted by LogBERT's preprocessing scripts.
    """

    sequentials: List[List[Union[int, str]]]
    quantitatives: Optional[np.ndarray] = None
    semantics: Optional[np.ndarray] = None
    parameters: Optional[np.ndarray] = None
    times: Optional[List[List[float]]] = None
    labels: Optional[np.ndarray] = None

    @property
    def n_sequences(self) -> int:
        return len(self.sequentials)

    def empty(self) -> bool:
        return self.n_sequences == 0


def _normalize_token(token: Any) -> Union[int, str]:
    if isinstance(token, (int, np.integer)):
        return int(token)
    if isinstance(token, bytes):
        token = token.decode("utf-8", errors="ignore")
    if isinstance(token, str):
        stripped = token.strip()
        if not stripped:
            return ""
        try:
            return int(stripped)
        except ValueError:
            return stripped
    if token is None:
        return ""
    token_str = str(token)
    try:
        return int(token_str)
    except ValueError:
        return token_str


def _normalize_sequence(seq: Sequence[Any]) -> List[Union[int, str]]:
    return [_normalize_token(tok) for tok in seq]


def _ensure_list_of_lists(value: Any) -> List[List[Union[int, str]]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        # object dtype from np.save can store lists directly
        if value.dtype == object:
            return [_normalize_sequence(seq) for seq in value]
        return [_normalize_sequence(seq.tolist()) for seq in value]
    if isinstance(value, (list, tuple)):
        return [_normalize_sequence(seq) for seq in value]
    raise TypeError(f"Unsupported Sequentials container type: {type(value)}")


def _optional_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    raise TypeError(f"Unsupported numeric container type: {type(value)}")


def _load_json_split(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        text = fh.read().strip()
        if not text:
            return {}
        # Accept either a plain JSON object or JSON Lines.
        if text[0] == "{":
            return json.loads(text)
        records = [json.loads(line) for line in text.splitlines() if line]
        if not records:
            return {}
        keys = records[0].keys()
        stacked = {k: [rec.get(k) for rec in records] for k in keys}
        return stacked


def fixed_window(line, window_size, adaptive_window, seq_len=None, min_len=0):
    line = [ln.split(",") for ln in line.split()]

    if len(line) < min_len:
        return [], []

    if seq_len is not None:
        line = line[:seq_len]

    if adaptive_window:
        window_size = len(line)

    line = np.array(line)

    if line.size == 0:
        return [], []

    if line.ndim == 1:
        line = line[:, np.newaxis]

    if line.shape[1] == 2:
        tim = line[:, 1].astype(float)
        log_tokens = line[:, 0]
        tim[0] = 0
    else:
        log_tokens = line[:, 0] if line.shape[1] > 0 else line.squeeze()
        tim = np.zeros(log_tokens.shape, dtype=float)

    logkey_seqs = []
    time_seq = []
    for i in range(0, len(log_tokens), max(window_size, 1)):
        logkey_seqs.append(log_tokens[i : i + window_size])
        time_seq.append(tim[i : i + window_size])

    return logkey_seqs, time_seq


def _diverse_subsample(
    logkey_trainset: np.ndarray,
    time_trainset: np.ndarray,
    train_sample: float,
    diverse_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample training sequences to a target fraction, mixing "diverse" and
    "non-diverse" examples.

    We first rank sequences by the number of distinct log keys (diversity).
    A fraction `diverse_ratio` of the target subset is sampled with
    probability proportional to this diversity score. The remaining
    (1 - diverse_ratio) fraction is sampled from the complement using
    the inverse diversity (favouring sequences with fewer distinct keys).
    """
    if train_sample >= 1.0:
        return logkey_trainset, time_trainset

    n = len(logkey_trainset)
    target = max(1, int(n * train_sample))
    if target >= n or n == 0:
        return logkey_trainset, time_trainset

    diverse_ratio = float(diverse_ratio)
    diverse_ratio = min(max(diverse_ratio, 0.0), 1.0)

    # Diversity score: number of unique tokens in each sequence
    uniq_counts = np.array([len(set(seq)) for seq in logkey_trainset], dtype=float)
    uniq_counts[uniq_counts <= 0] = 1.0

    # Normalised diversity weights
    w_div = uniq_counts / uniq_counts.sum()

    # Non-diverse weights: invert the ordering
    inv = uniq_counts.max() - uniq_counts
    if np.all(inv == 0):
        w_nondiv = np.ones_like(inv) / len(inv)
    else:
        w_nondiv = inv / inv.sum()

    rng = np.random.default_rng(seed)

    n_diverse = int(round(target * diverse_ratio))
    n_nondiv = target - n_diverse

    # Sample diverse subset
    idx_all = np.arange(n)
    if n_diverse > 0:
        idx_div = rng.choice(idx_all, size=n_diverse, replace=False, p=w_div)
    else:
        idx_div = np.array([], dtype=int)

    # Sample non-diverse subset from remaining indices
    if n_nondiv > 0:
        mask_remaining = np.ones(n, dtype=bool)
        mask_remaining[idx_div] = False
        remaining = idx_all[mask_remaining]
        if remaining.size > 0:
            # Re-normalise non-diverse weights over remaining pool
            w_nondiv_rem = w_nondiv[remaining]
            if w_nondiv_rem.sum() == 0:
                w_nondiv_rem[:] = 1.0 / len(w_nondiv_rem)
            else:
                w_nondiv_rem /= w_nondiv_rem.sum()
            size = min(n_nondiv, remaining.size)
            idx_nondiv = rng.choice(remaining, size=size, replace=False, p=w_nondiv_rem)
        else:
            idx_nondiv = np.array([], dtype=int)
    else:
        idx_nondiv = np.array([], dtype=int)

    idx = np.concatenate([idx_div, idx_nondiv])
    if idx.size == 0:
        return logkey_trainset, time_trainset
    return logkey_trainset[idx], time_trainset[idx]


def generate_train_valid(
    data_path,
    window_size=20,
    adaptive_window=True,
    sample_ratio=1,
    valid_size=0.1,
    output_path=None,
    scale=None,
    scale_path=None,
    seq_len=None,
    min_len=0,
    train_sample=1.
):
    with open(data_path, "r", encoding="utf-8") as f:
        data_iter = f.readlines()
    
    num_session = int(len(data_iter) * sample_ratio)
    test_size = int(min(num_session, len(data_iter)) * valid_size)

    print("before filtering short session")
    print("train size ", int(num_session - test_size))
    print("valid size ", int(test_size))
    print("=" * 40)

    logkey_seq_pairs = []
    time_seq_pairs = []
    session = 0
    for line in data_iter:
        if session >= num_session:
            break
        session += 1

        logkeys, times = fixed_window(line, window_size, adaptive_window, seq_len, min_len)
        logkey_seq_pairs += logkeys
        time_seq_pairs += times

    logkey_seq_pairs = np.array(logkey_seq_pairs, dtype=object)
    time_seq_pairs = np.array(time_seq_pairs, dtype=object)

    if len(logkey_seq_pairs) == 0:
        raise ValueError(f"No sequences generated from {data_path}.")

    logkey_trainset, logkey_validset, time_trainset, time_validset = train_test_split(
        logkey_seq_pairs,
        time_seq_pairs,
        test_size=test_size,
        random_state=1234,
    )

    train_len = list(map(len, logkey_trainset))
    valid_len = list(map(len, logkey_validset))

    train_sort_index = np.argsort(-1 * np.array(train_len))
    valid_sort_index = np.argsort(-1 * np.array(valid_len))

    logkey_trainset = logkey_trainset[train_sort_index]
    logkey_validset = logkey_validset[valid_sort_index]

    time_trainset = time_trainset[train_sort_index]
    time_validset = time_validset[valid_sort_index]

    logkey_trainset, time_trainset = _diverse_subsample(
        logkey_trainset,
        time_trainset,
        train_sample=train_sample,
        diverse_ratio=0.5,
        seed=42,
    )

    print("=" * 40)
    print("Num of train seqs", len(logkey_trainset))
    print("Num of valid seqs", len(logkey_validset))
    print("=" * 40)

    return logkey_trainset, logkey_validset, time_trainset, time_validset


def generate_train_valid_sessions(
    train_path: os.PathLike[str] | str,
    *,
    window_size: int,
    adaptive_window: bool,
    sample_ratio: float,
    valid_ratio: float,
    seq_len: Optional[int],
    min_len: int,
    rng: np.random.Generator,
    train_sample: float = 1.
) -> Tuple[List[List[int]], List[List[int]], List[List[float]], List[List[float]]]:
    _ = rng  # RNG retained for signature compatibility
    log_trainset, log_validset, time_trainset, time_validset = generate_train_valid(
        train_path,
        window_size=window_size,
        adaptive_window=adaptive_window,
        sample_ratio=sample_ratio,
        valid_size=valid_ratio,
        seq_len=seq_len,
        min_len=min_len,
        train_sample=train_sample
    )

    def _to_token_lists(array: np.ndarray) -> List[List[Union[int, str]]]:
        return [_normalize_sequence(seq) for seq in array.tolist()]

    def _to_float_lists(array: np.ndarray) -> List[List[float]]:
        out: List[List[float]] = []
        for seq in array.tolist():
            out.append([float(x) for x in seq])
        return out

    train_logs = _to_token_lists(log_trainset)
    valid_logs = _to_token_lists(log_validset)
    train_times = _to_float_lists(time_trainset)
    valid_times = _to_float_lists(time_validset)

    return train_logs, valid_logs, train_times, valid_times


def load_windowed_split(
    path: os.PathLike[str] | str,
    *,
    window_size: int,
    adaptive_window: bool,
    seq_len: Optional[int],
    min_len: int,
    label: Optional[int] = None,
) -> LogBERTSplit:
    log_windows: List[List[int]] = []
    time_windows: List[List[float]] = []

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            logs, times = fixed_window(line, window_size, adaptive_window, seq_len, min_len)
            for seq_logs, seq_times in zip(logs, times):
                seq_logs = _normalize_sequence(seq_logs)
                seq_times = [float(t) for t in seq_times]
                if not seq_logs:
                    continue
                log_windows.append(seq_logs)
                time_windows.append(seq_times)

    labels = [label] * len(log_windows) if label is not None else None
    return build_split(log_windows, times=time_windows, labels=labels)


def build_split(
    sequences: Sequence[Sequence[Any]],
    *,
    times: Optional[Sequence[Sequence[float]]] = None,
    labels: Optional[Sequence[int]] = None,
) -> LogBERTSplit:
    seq_list = [_normalize_sequence(seq) for seq in sequences]
    time_list: Optional[List[List[float]]] = None
    if times is not None:
        time_list = [list(map(float, seq)) for seq in times]
    if labels is not None:
        labels_arr = np.asarray(labels, dtype=np.int64)
    else:
        labels_arr = None
    return LogBERTSplit(sequentials=seq_list, times=time_list, labels=labels_arr)


def concatenate_splits(splits: Sequence[LogBERTSplit]) -> LogBERTSplit:
    sequences: List[List[int]] = []
    times: List[List[float]] = []
    labels: List[int] = []
    for split in splits:
        sequences.extend(split.sequentials)
        if split.times is not None:
            times.extend(split.times)
        elif split.sequentials:
            times.extend([[0.0] * len(seq) for seq in split.sequentials])
        if split.labels is not None:
            labels.extend(split.labels.tolist())
    times_list = times if times else None
    labels_arr = np.asarray(labels, dtype=np.int64) if labels else None
    return LogBERTSplit(sequentials=sequences, times=times_list, labels=labels_arr)


def load_logbert_split(path: os.PathLike[str] | str) -> LogBERTSplit:
    """
    Load a processed LogBERT artifact (npz, pkl, json/jsonl) into a unified structure.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    suffix = file_path.suffix.lower()
    if suffix == ".npz":
        raw = np.load(file_path, allow_pickle=True)
        data = {k: raw[k] for k in raw.files}
    elif suffix in (".pkl", ".pickle", ".pk"):
        with file_path.open("rb") as fh:
            data = pickle.load(fh)
    elif suffix in (".json", ".jsonl"):
        data = _load_json_split(file_path)
    elif suffix in ("", ".txt"):
        sequentials: List[List[int]] = []
        times: List[List[float]] = []
        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                log_windows, time_windows = fixed_window(line, window_size=1, adaptive_window=True)
                for seq_logs, seq_times in zip(log_windows, time_windows):
                    seq_logs = _normalize_sequence(seq_logs)
                    seq_times = [float(t) for t in seq_times]
                    if seq_logs:
                        sequentials.append(seq_logs)
                        times.append(seq_times)
        return LogBERTSplit(sequentials=sequentials, times=times)
    else:
        raise ValueError(f"Unsupported artifact format: {file_path.suffix}")

    times = None
    if isinstance(data, dict):
        sequentials = _ensure_list_of_lists(data.get("Sequentials"))
        quantitatives = _optional_array(data.get("Quantitatives"))
        semantics = _optional_array(data.get("Semantics"))
        parameters = _optional_array(data.get("Parameters"))
        times = data.get("Times")
        if times is not None:
            times = [list(map(float, seq)) for seq in times]
        labels = data.get("labels") or data.get("Labels")
    elif isinstance(data, (list, tuple)):
        if not data:
            sequentials, quantitatives, semantics, parameters, times, labels = [], None, None, None, None, None
        else:
            sequentials = _ensure_list_of_lists([item.get("Sequentials", []) for item in data])
            quantitatives = _optional_array([item.get("Quantitatives") for item in data])
            semantics = _optional_array([item.get("Semantics") for item in data])
            parameters = _optional_array([item.get("Parameters") for item in data])
            times = [item.get("Times") for item in data]
            times = [list(map(float, seq)) if seq is not None else [0.0] * len(seq2) for seq, seq2 in zip(times, sequentials)]
            labels = np.asarray([item.get("labels") for item in data], dtype=np.int64)
    else:
        raise TypeError(f"Unexpected artifact payload type: {type(data)}")

    if labels is not None and not isinstance(labels, np.ndarray):
        labels = np.asarray(labels, dtype=np.int64)

    return LogBERTSplit(
        sequentials=sequentials,
        quantitatives=quantitatives,
        semantics=semantics,
        parameters=parameters,
        times=times,
        labels=labels,
    )


def load_vocab(path: os.PathLike[str] | str):
    vocab_path = Path(path)
    if not vocab_path.exists():
        raise FileNotFoundError(vocab_path)
    suffix = vocab_path.suffix.lower()
    if suffix in (".pkl", ".pickle", ".pk"):
        return WordVocab.load_vocab(str(vocab_path))
    if suffix in (".json", ".js"):
        with vocab_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    raise ValueError(f"Unsupported vocab file format: {suffix}")


def infer_vocab_size(sequentials: Sequence[Sequence[int]], reserved: int = 0) -> int:
    vmax = 0
    for seq in sequentials:
        if not seq:
            continue
        vmax = max(vmax, max(seq))
    return int(vmax + 1 + reserved)


def determine_vocab_size(vocab_object: Any, sequentials: Sequence[Sequence[int]], reserved: int = 0) -> int:
    vocab_size = 0
    if vocab_object is not None:
        try:
            size = len(vocab_object)
            if size:
                vocab_size = int(size)
        except TypeError:
            pass
        if isinstance(vocab_object, dict) and vocab_object:
            vmax = max(vocab_object.values())
            vocab_size = int(vmax + 1)
    else:
        vocab_size = infer_vocab_size(sequentials, reserved)
    return vocab_size  # max(vocab_size, data_size)


def build_token_mapper(vocab_object: Any) -> Optional[Callable[[Union[int, str]], int]]:
    if vocab_object is None:
        return None

    stoi: Optional[Dict[str, int]] = None
    unk_index = 0

    if hasattr(vocab_object, "stoi"):
        stoi = {str(k): int(v) for k, v in vocab_object.stoi.items()}
        unk_index = int(getattr(vocab_object, "unk_index", 1))
    elif isinstance(vocab_object, dict):
        candidate = vocab_object.get("stoi") if isinstance(vocab_object.get("stoi"), dict) else vocab_object
        if isinstance(candidate, dict):
            stoi = {str(k): int(v) for k, v in candidate.items() if isinstance(v, (int, np.integer))}
            if "unk_index" in vocab_object:
                unk_index = int(vocab_object["unk_index"])
            else:
                unk_index = int(max(stoi.values(), default=0) + 1)

    if not stoi:
        return None

    def mapper(token: Union[int, str]) -> int:
        if isinstance(token, (int, np.integer)):
            return int(token)
        token_str = "" if token is None else str(token).strip()
        if not token_str:
            return unk_index
        try:
            return int(token_str)
        except (ValueError, TypeError):
            return stoi.get(token_str, unk_index)

    return mapper


def encode_split_sequences(split: Optional[LogBERTSplit], mapper: Optional[Callable[[Union[int, str]], int]]) -> None:
    if split is None or mapper is None or not split.sequentials:
        return
    needs_mapping = any(
        any(not isinstance(tok, (int, np.integer)) for tok in seq)
        for seq in split.sequentials
    )
    if not needs_mapping:
        return
    for seq in split.sequentials:
        for idx, tok in enumerate(seq):
            if isinstance(tok, (int, np.integer)):
                continue
            seq[idx] = mapper(tok)


def bag_of_tokens(indices: Sequence[int], vocab_size: int) -> np.ndarray:
    """
    Dense bag-of-tokens feature with explicit range enforcement on indices.

    To avoid accidental huge allocations from stray large ids, we
    explicitly clamp the index set to [0, vocab_size) before calling
    np.bincount. Any token outside the vocabulary range is ignored,
    effectively behaving like an <unk> bucket that is not counted.
    """
    if vocab_size <= 0:
        return np.zeros(0, dtype=np.float32)
    if not indices:
        return np.zeros(vocab_size, dtype=np.float32)

    idx = np.asarray(indices, dtype=np.int64)
    # Keep only valid ids; drop anything outside the vocab range.
    valid = (idx >= 0) & (idx < vocab_size)
    if not np.any(valid):
        return np.zeros(vocab_size, dtype=np.float32)
    idx = idx[valid]

    counts = np.bincount(idx, minlength=vocab_size)
    return counts.astype(np.float32, copy=False)


def compute_sequence_features(
    split: LogBERTSplit,
    vocab_size: int,
    *,
    use_sequentials: bool = True,
    use_quantitatives: bool = True,
    use_semantics: bool = False,
    use_parameters: bool = False,
) -> np.ndarray:
    """
    Build per-sequence static features used by reconstruction/quantitative HKAN heads.
    """
    features: List[np.ndarray] = []
    for idx, seq in enumerate(split.sequentials):
        parts: List[np.ndarray] = []
        if use_sequentials:
            parts.append(bag_of_tokens(seq, vocab_size))
        if use_quantitatives and split.quantitatives is not None:
            parts.append(split.quantitatives[idx].astype(np.float32, copy=False))
        if use_semantics and split.semantics is not None:
            parts.append(split.semantics[idx].astype(np.float32, copy=False))
        if use_parameters and split.parameters is not None:
            parts.append(split.parameters[idx].astype(np.float32, copy=False))
        if not parts:
            raise ValueError("At least one feature group must be selected.")
        features.append(np.concatenate(parts, dtype=np.float32))
    return np.vstack(features) if features else np.zeros((0, 0), dtype=np.float32)


def make_masked_contexts(
    split: LogBERTSplit,
    vocab_size: int,
    *,
    window_left: int = 5,
    window_right: int = 5,
    mask_probability: float = 0.15,
    max_masks_per_sequence: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    include_quantitatives: bool = False,
    include_semantics: bool = False,
    include_parameters: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce masked-token training/evaluation examples mirroring LogBERT's MLM task.

    Returns
    -------
    X : np.ndarray [N, D]
        Context features for HKAN input.
    y : np.ndarray [N]
        Target token ids.
    seq_index : np.ndarray [N]
        Index of the originating sequence for aggregation.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    contexts: List[np.ndarray] = []
    targets: List[int] = []
    seq_ids: List[int] = []

    for seq_idx, seq in enumerate(split.sequentials):
        if not seq:
            continue
        seq_len = len(seq)
        candidate_positions = list(range(seq_len))
        if mask_probability < 1.0:
            mask_flags = rng.random(seq_len) < mask_probability
            candidate_positions = [pos for pos, flag in enumerate(mask_flags) if flag]
        if max_masks_per_sequence is not None and len(candidate_positions) > max_masks_per_sequence:
            rng.shuffle(candidate_positions)
            candidate_positions = candidate_positions[:max_masks_per_sequence]

        for pos in candidate_positions:
            token_id = int(seq[pos])
            left_ctx = seq[max(0, pos - window_left) : pos]
            right_ctx = seq[pos + 1 : pos + 1 + window_right]

            feat_parts: List[np.ndarray] = [
                bag_of_tokens(left_ctx, vocab_size),
                bag_of_tokens(right_ctx, vocab_size),
            ]
            if include_quantitatives and split.quantitatives is not None:
                feat_parts.append(split.quantitatives[seq_idx].astype(np.float32, copy=False))
            if include_semantics and split.semantics is not None:
                feat_parts.append(split.semantics[seq_idx].astype(np.float32, copy=False))
            if include_parameters and split.parameters is not None:
                feat_parts.append(split.parameters[seq_idx].astype(np.float32, copy=False))

            contexts.append(np.concatenate(feat_parts, dtype=np.float32))
            targets.append(token_id)
            seq_ids.append(seq_idx)

    if not contexts:
        return (
            np.zeros((0, vocab_size * 2), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )

    X = np.vstack(contexts)  # pad_and_stack(contexts))
    y = np.asarray(targets, dtype=np.int64)
    seq_index = np.asarray(seq_ids, dtype=np.int64)
    return X, y, seq_index


def pad_and_stack(arrays, pad_value=0):
    """
    Pads a list or array of 1D/2D numpy arrays so they all have equal length
    along their last dimension, then stacks them vertically.

    Parameters
    ----------
    arrays : list or np.ndarray
        A list (or array) of numpy arrays, e.g. shape (N,) or (N, variable_length)
    pad_value : scalar, optional
        Value to pad with. Default is 0.

    Returns
    -------
    stacked : np.ndarray
        A 2D or 3D numpy array with equal-length subarrays.
    """

    # Ensure it's a list of np.ndarrays
    arrays = [np.asarray(a) for a in arrays]

    # Find maximum length along the last axis
    max_len = max(a.shape[-1] for a in arrays)

    # Pad each array along its last axis
    padded = [
        np.pad(a, 
               [(0, 0)] * (a.ndim - 1) + [(0, max_len - a.shape[-1])],
               mode='constant', 
               constant_values=pad_value)
        for a in arrays
    ]
    
    return padded


def gather_unique_labels(split: LogBERTSplit) -> np.ndarray:
    if split.labels is None:
        return np.zeros(split.n_sequences, dtype=np.int64)
    return split.labels.astype(np.int64, copy=False)
