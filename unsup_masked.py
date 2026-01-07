import numpy as np
from hkan_unsup import MultiHKANRegressor
from tqdm import tqdm
from log import logger
from pathlib import Path
import joblib


class HKANMaskedLanguageModel:
    """
    HKAN analogue of LogBERT's masked language modeling head.
    """

    def __init__(self, hkan_factory, vocab_size, *, epsilon=1e-8, temperature=1.0, linear_factory=None):
        self.hkan_factory = hkan_factory
        self.vocab_size = int(vocab_size)
        self.epsilon = float(epsilon)
        self.temperature = float(temperature)
        self.reg = None
        self.linear_factory = linear_factory
        self.chunk_temperatures = None
        self.calibration_chunk_size = None
        self.calibration_mode = "global"

    def save(self, path):
        """
        Serialize the trained MLM to disk.

        Parameters
        ----------
        path : str or Path
            Destination file (e.g. 'runs/exp1/mlm_model.joblib').
        """
        path = Path(path)
        # Note: we intentionally do NOT persist `linear_factory` or the original
        # `hkan_factory` here, because they may be non-picklable closures
        # (e.g. lambdas defined in the training script). For inference and
        # interpretability we only need the fitted encoder + linear head and
        # the calibration state.
        if self.reg is not None:
            # Create a lightweight, picklable MultiHKANRegressor that carries
            # only the trained encoder_ and linear_model_.
            stripped_reg = MultiHKANRegressor(hkan_factory=None)
            stripped_reg.encoder_ = self.reg.encoder_
            stripped_reg.linear_model_ = self.reg.linear_model_
        else:
            stripped_reg = None

        obj = {
            "vocab_size": self.vocab_size,
            "epsilon": self.epsilon,
            "temperature": self.temperature,
            "reg": stripped_reg,
            "chunk_temperatures": self.chunk_temperatures,
            "calibration_chunk_size": self.calibration_chunk_size,
            "calibration_mode": self.calibration_mode,
        }
        joblib.dump(obj, path)

    @classmethod
    def load(cls, path, hkan_factory):
        """
        Reload a previously saved MLM for inference.

        Parameters
        ----------
        path : str or Path
            Path to the .joblib file saved by `save`.
        hkan_factory : callable
            The same HKAN factory used at training time.

        Returns
        -------
        HKANMaskedLanguageModel
            A model ready for `predict_proba`, `token_in_topk`, etc.
        """
        path = Path(path)
        data = joblib.load(path)
        model = cls(
            hkan_factory,
            data["vocab_size"],
            epsilon=data.get("epsilon", 1e-8),
            temperature=data.get("temperature", 1.0),
        )
        model.reg = data["reg"]
        model.chunk_temperatures = data["chunk_temperatures"]
        model.calibration_chunk_size = data["calibration_chunk_size"]
        model.calibration_mode = data["calibration_mode"]
        return model

    def _sanitize_targets(self, y: np.ndarray) -> np.ndarray:
        targets = np.asarray(y, dtype=np.int64)
        if targets.size == 0:
            return targets
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        # Map any out-of-range id into an <unk> bucket
        unk_idx = self.vocab_size - 1
        bad = (targets < 0) | (targets >= self.vocab_size)
        if np.any(bad):
            n_bad = int(bad.sum())
            # logger.warning(
            #     f"HKANMaskedLanguageModel: {n_bad} labels outside [0, {self.vocab_size - 1}); mapping them to <unk> index {unk_idx}",
            # )
            targets = targets.copy()
            targets[targets < 0] = unk_idx
            targets[targets >= self.vocab_size] = unk_idx
        return targets

    def fit(self, X, y, calibrate_temperature=True):
        n_samples = X.shape[0]
        logger.info(
            f"HKANMaskedLanguageModel: fitting on {n_samples} masked samples (vocab_size={self.vocab_size})",            
        )
        if n_samples != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        import time
        t0 = time.perf_counter()

        # 1) Prepare targets
        logger.info("HKANMaskedLanguageModel: constructing targets array")
        targets = np.asarray(y, dtype=np.int64)
        if targets.size == 0:
            raise ValueError("Cannot fit MLM with zero targets.")
        # Optional safety: clip out-of-range labels into an <unk> bucket
        if targets.min() < 0 or targets.max() >= self.vocab_size:
            unk_idx = self.vocab_size - 1
            bad_low = targets < 0
            bad_high = targets >= self.vocab_size
            n_bad = int(bad_low.sum() + bad_high.sum())
            # logger.warning(
            #     f"HKANMaskedLanguageModel: {n_bad} labels outside [0, {self.vocab_size - 1}); mapping them to <unk> index {unk_idx}",
            # )
            targets = targets.copy()
            targets[bad_low] = unk_idx
            targets[bad_high] = unk_idx

        # 2) Build one-hot matrix
        logger.info(
            f"HKANMaskedLanguageModel: building one-hot targets of shape ({n_samples}, {self.vocab_size})",            
        )
        targets = self._sanitize_targets(targets)
        one_hot = np.zeros((n_samples, self.vocab_size), dtype=np.float32)
        rows = np.arange(n_samples, dtype=np.int64)
        one_hot[rows, targets] = 1.0
        logger.info(
            f"HKANMaskedLanguageModel: one-hot construction done in {round(time.perf_counter() - t0, 4)}s",            
        )

        # 3) Fit the HKAN regressor
        logger.info(
            f"HKANMaskedLanguageModel: fitting MultiHKANRegressor head with {self.vocab_size} outputs"
        )
        t1 = time.perf_counter()
        self.reg = MultiHKANRegressor(self.hkan_factory, linear_factory=self.linear_factory, logger=logger)
        self.reg.fit(X, one_hot)
        logger.info(
            f"HKANMaskedLanguageModel: regressor fit complete in {time.perf_counter() - t1} s",            
        )

        # 4) Optional temperature calibration
        if calibrate_temperature:
            logger.info("HKANMaskedLanguageModel: calibrating temperature")
            token_loss = self.token_neg_log_likelihood(X, targets, temperature=1.0)
            median_loss = float(np.median(token_loss))
            self.temperature = max(median_loss, self.epsilon)
            self.chunk_temperatures = None
            self.calibration_chunk_size = None
            logger.info(
                "HKANMaskedLanguageModel: temperature calibrated to %.3f "
                "(overall fitting time: %.1f s)",
                self.temperature,
                time.perf_counter() - t0,
            )
        else:
            logger.info(
                f"HKANMaskedLanguageModel: fit complete without temperature calibration (overall time: {time.perf_counter() - t0}s)",
            )

        return self

    def _raw_scores(self, X, mode='CPU'):
        if self.reg is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.reg.predict(X, mode=mode)

    def _temperature_for_chunk(self, chunk_index):
        if self.chunk_temperatures is None or chunk_index is None:
            return max(self.temperature, self.epsilon)
        idx = min(max(int(chunk_index), 0), len(self.chunk_temperatures) - 1)
        return max(float(self.chunk_temperatures[idx]), self.epsilon)

    def _scaled_logits(self, X, *, mode='CPU', temperature=None, chunk_index=None):
        logits = self._raw_scores(X, mode=mode)
        temp = temperature if temperature is not None else self._temperature_for_chunk(chunk_index)
        return logits / max(temp, self.epsilon)

    def predict_proba(self, X, *, mode='CPU', temperature=None, chunk_index=None):
        logits = self._scaled_logits(X, mode=mode, temperature=temperature, chunk_index=chunk_index)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        denom = exp_logits.sum(axis=1, keepdims=True) + self.epsilon
        return exp_logits / denom

    def token_neg_log_likelihood(self, X, targets, *, mode='CPU', temperature=None, chunk_index=None):
        targets = self._sanitize_targets(targets)
        probs = self.predict_proba(X, mode=mode, temperature=temperature, chunk_index=chunk_index)
        probs = probs[np.arange(len(targets)), targets]
        probs = np.clip(probs, self.epsilon, 1.0)
        return -np.log(probs)

    def _topk_indices(self, logits, topk, *, sort=True):
        n_samples, vocab_size = logits.shape
        topk = int(topk)
        if topk <= 0:
            raise ValueError("topk must be positive.")
        topk = min(topk, vocab_size)
        if topk == 0:
            return np.zeros((n_samples, 0), dtype=np.int64)

        if topk == vocab_size:
            indices = np.tile(np.arange(vocab_size), (n_samples, 1))
            if sort:
                return np.argsort(-logits, axis=1)
            return indices

        if topk == 1:
            return np.argmax(logits, axis=1, keepdims=True)

        kth = vocab_size - topk
        partition = np.argpartition(logits, kth=kth, axis=1)[:, kth:]
        if sort:
            rows = np.arange(n_samples)[:, None]
            order = np.argsort(-logits[rows, partition], axis=1)
            partition = partition[rows, order]
        return partition

    def predict_topk(self, X, topk=1, *, mode='CPU', chunk_index=None):
        """
        1. _scaled_logits reuses the same temperature-adjusted matrix.
        2. _topk_indices grabs the top‑k indices via np.argpartition (no full sort, no exponentials). It also short-circuits the k=1 and k >= vocab cases.
        3. predict_topk delegates to _topk_indices, and token_in_topk uses the same helper without sorting, returning membership with a simple comparison.

        This avoids building the entire probability matrix and heavy sorting, giving a big speedup on large batches while keeping behaviour identical.

        Note: Previously, the function built a full softmax matrix (np.exp, np.argsort) every time. The top‑K path was rewritten so it operates directly on the pre-softmax logits
        Args:
            X (_type_): _description_
            topk (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        logits = self._scaled_logits(X, mode=mode, chunk_index=chunk_index)
        if logits.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.int64)
        return self._topk_indices(logits, topk, sort=True)

    def token_in_topk(self, X, targets, topk=1, mode='CPU', chunk_index=None):
        targets = self._sanitize_targets(targets)
        targets = np.asarray(targets, dtype=np.int64)
        if targets.size == 0:
            return np.zeros(0, dtype=bool)
        logits = self._scaled_logits(X, mode=mode, chunk_index=chunk_index)
        if logits.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        topk = int(topk)
        if topk >= logits.shape[1]:
            return np.ones_like(targets, dtype=bool)
        idx = self._topk_indices(logits, topk, sort=False)
        targets = targets.reshape(-1, 1)
        return np.any(idx == targets, axis=1)

    def calibrate_temperature(self, X, targets, *, mode="global", chunk_size=100, device_mode="CPU"):
        """
        Calibrate temperature scalars using held-out validation tokens.

        mode
            "global": single scalar (median NLL, legacy behaviour)
            "chunk": per-chunk medians retained and applied during batched inference
        """
        if X.shape[0] == 0:
            logger.warning(
                f"HKANMaskedLanguageModel: calibration skipped (no samples); keeping temperature={round(self.temperature, 3)}",
            )
            return self

        chunk_size = int(chunk_size) if chunk_size else X.shape[0]
        chunk_size = max(chunk_size, 1)

        if mode == "global":
            losses = self.token_neg_log_likelihood(X, targets, mode=device_mode, temperature=1.0)
            self.temperature = max(float(np.median(losses)), self.epsilon)
            self.chunk_temperatures = None
            self.calibration_chunk_size = None
            self.calibration_mode = "global"
            logger.info(
                f"HKANMaskedLanguageModel: global calibration temperature={round(self.temperature, 3)}"
            )
            return self

        if mode != "chunk":
            raise ValueError(f"Unsupported calibration mode '{mode}'")

        temps = []
        n = X.shape[0]
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            chunk_losses = self.token_neg_log_likelihood(
                X[start:stop],
                targets[start:stop],
                mode=device_mode,
                temperature=1.0,
            )
            temps.append(max(float(np.median(chunk_losses)), self.epsilon))

        self.chunk_temperatures = np.asarray(temps, dtype=np.float32)
        self.calibration_chunk_size = chunk_size
        self.temperature = float(np.median(self.chunk_temperatures))
        self.calibration_mode = "chunk"
        logger.info(
            f"HKANMaskedLanguageModel: chunk calibration stored {len(self.chunk_temperatures)} temps "
            f"(median={round(self.temperature, 3)}, min={round(float(self.chunk_temperatures.min()),3)}, max={round(float(self.chunk_temperatures.max()), 3)})",
        )
        return self

    def score_sequences(self, X, targets, seq_index, agg="mean", total_sequences=None):
        token_scores = self.token_neg_log_likelihood(X, targets)
        if len(token_scores) == 0:
            total = 0 if total_sequences is None else int(total_sequences)
            return np.zeros(total, dtype=np.float32)

        seq_index = np.asarray(seq_index, dtype=np.int64)
        n_seqs = int(seq_index.max() + 1) if total_sequences is None else int(total_sequences)
        aggregated = np.zeros(n_seqs, dtype=np.float32)

        if agg == "mean":
            counts = np.zeros(n_seqs, dtype=np.int64)
            for score, idx in zip(token_scores, seq_index):
                aggregated[idx] += score
                counts[idx] += 1
            counts = np.maximum(counts, 1)
            aggregated /= counts
        elif agg == "max":
            aggregated.fill(-np.inf)
            for score, idx in zip(token_scores, seq_index):
                aggregated[idx] = max(aggregated[idx], score)
            aggregated[aggregated == -np.inf] = 0.0
        else:
            raise ValueError(f"Unsupported aggregation: {agg}")

        return aggregated
