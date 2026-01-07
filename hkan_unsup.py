import re
import tqdm
import threading
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import clone
from dataclasses import dataclass
from collections import Counter
from joblib import Parallel, delayed
from parallellisation import tqdm_joblib


class LinearHead:
    """
    Simple, picklable linear head: y = Z @ coef_.

    This is used by MultiHKANRegressor instead of a closure-defined inner
    class so that saved models (via joblib) remain serialisable.
    """

    def __init__(self, coef_: np.ndarray):
        self.coef_ = np.asarray(coef_)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.coef_


class MultiHKANRegressor:
    """
    Shared HKAN encoder with a multi-output linear head.
    """

    def __init__(self, hkan_factory, logger=None, linear_factory=None):
        self.hkan_factory = hkan_factory
        self.linear_factory = linear_factory or LinearRegression
        self.encoder_ = None
        self.linear_model_ = None
        self.logger = logger

    @staticmethod
    def _flatten_features(features: np.ndarray) -> np.ndarray:
        if features.ndim != 3:
            raise ValueError(f"Expected HKAN features with 3 dims, got {features.shape}")
        # (n_vars_out, n_vars_in, n_samples) -> (n_samples, n_vars_out * n_vars_in)
        return np.transpose(features, (2, 0, 1)).reshape(features.shape[2], -1)

    @staticmethod
    def _extract_expanding_layer(model):
        if not hasattr(model, "steps"):
            raise ValueError("HKAN factory must return a pipeline with named steps.")
        for name, step in model.steps:
            if name.startswith("expanding_layer"):
                return step
        raise ValueError("No expanding_layer step found in HKAN pipeline.")

    def fit(self, X, Y):
        def _batch_gram(X_b, Y_b, encoder, mode):
            feats_b = self._flatten_features(encoder.transform(X_b, mode=mode))
            G_b = feats_b.T @ feats_b
            H_b = feats_b.T @ Y_b
            return G_b, H_b
        
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        pipeline = self.hkan_factory()
        if self.logger is not None:
            self.logger.info("Fitting on a single label to determine feature dimension")
        expanding = self._extract_expanding_layer(pipeline)
        expanding.fit(X, Y[:, 0])
        # Streaming linear regression using normal equations:
        # we avoid materialising the full feature matrix for all samples.
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("MultiHKANRegressor.fit received empty X.")

        # Obtain a first batch of features to determine feature dimension.
        batch_size = getattr(expanding, "batch_size", None) or n_samples
        batch_size = max(1, min(batch_size, n_samples))
        first_stop = min(batch_size, n_samples)
        feats0 = self._flatten_features(expanding.transform(X[:first_stop]))
        n_features = feats0.shape[1]
        n_outputs = Y.shape[1]

        # Accumulate normal equations: G = Z^T Z, H = Z^T Y
        G = feats0.T @ feats0  # [D, D]
        H = feats0.T @ Y[:first_stop]  # [D, K]

        # -- Stream ridge regression
        # Stream remaining batches with progress bar
        # Build remaining batch index ranges
        starts = list(range(first_stop, n_samples, batch_size))

        lock = threading.Lock()

        def _accumulate_batch(start_idx: int):
            """
            Worker: compute local G_b, H_b for batch, then add to shared G, H.
            """
            stop_idx = min(start_idx + batch_size, n_samples)
            X_b = X[start_idx:stop_idx]
            Y_b = Y[start_idx:stop_idx]
            feats_b = self._flatten_features(expanding.transform(X_b))

            G_b = feats_b.T @ feats_b
            H_b = feats_b.T @ Y_b

            # Add to shared matrices under lock to avoid race conditions
            with lock:
                G[:] += G_b
                H[:] += H_b

        if starts:
            with tqdm_joblib(
                tqdm.tqdm(
                    total=len(starts),
                    desc="Ridge reg. streaming batches",
                    mininterval=1.0,
                )
            ):
                Parallel(n_jobs=2, prefer="threads")(
                    delayed(_accumulate_batch)(start) for start in starts
                )
        # -- End stream ridge regression

        # Small ridge term for numerical stability
        lam = 1e-6
        G += lam * np.eye(n_features, dtype=G.dtype)
        coef = np.linalg.solve(G, H)  # [D, K]

        self.encoder_ = expanding
        self.linear_model_ = LinearHead(coef)
        return self

    def predict(self, X, mode='GPU', batch_size: int | None = None):
        """
        Predict Y for input X in a memory-efficient, batched fashion.

        Instead of materialising the full HKAN feature tensor for all samples
        at once (which would allocate an array of shape
        [n_vars_out, n_vars_in, n_samples]), we stream X in chunks:

            1. Run encoder_.transform on a chunk of X.
            2. Flatten features for that chunk only.
            3. Apply the linear head to get predictions.
            4. Discard intermediate features and move to the next chunk.

        Parameters
        ----------
        X : np.ndarray [N, P]
            Input feature matrix.
        mode : {'GPU', 'CPU'}
            Passed through to the underlying HKAN encoder transform.
        batch_size : int, optional
            Number of samples per prediction batch. If None, falls back to
            encoder_.batch_size or the full dataset size.
        """
        if self.encoder_ is None or self.linear_model_ is None:
            raise RuntimeError("MultiHKANRegressor must be fitted before predicting.")

        n_samples = X.shape[0]
        if n_samples == 0:
            return np.zeros((0, 1), dtype=float)

        # Derive a reasonable batch size
        if batch_size is None:
            batch_size = getattr(self.encoder_, "batch_size", None) or n_samples
        if batch_size <= 0:
            raise ValueError("batch_size must be positive in MultiHKANRegressor.predict.")

        preds = []
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            X_chunk = X[start:stop]
            feats_chunk = self._flatten_features(self.encoder_.transform(X_chunk, mode=mode))
            y_chunk = self.linear_model_.predict(feats_chunk)
            preds.append(np.asarray(y_chunk))

        return np.concatenate(preds, axis=0)
    