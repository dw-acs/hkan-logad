import itertools
import tqdm
import gc

import numpy as np
from joblib import Parallel, delayed
try:
    import cupy as cp  # GPU-accelerated NumPy
    CUPY_OK = True
except Exception:
    CUPY_OK = False

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from parallellisation import tqdm_joblib


MININTERVAL = 10
TQDM_DISABLE = False 


def set_mininterval(mininterval):
    global MININTERVAL
    MININTERVAL = mininterval


def set_tqdm_disable(disable):
    global TQDM_DISABLE
    TQDM_DISABLE = disable

class Sigmoid:
    def __init__(self, s=1):
        self.s = s

    def __call__(self, x):
        return 1 / (1 + np.exp(-self.s * x))

class Gaussian:
    def __init__(self, s=1):
        self.s = s

    def __call__(self, x):
        return np.exp(-((self.s * x) ** 2))

class ReLU:
    def __init__(self, s=1):
        self.s = s

    def __call__(self, x):
        return np.maximum(0, self.s * x)

class Tanh:
    def __init__(self, s=1):
        self.s = s

    def __call__(self, x):
        return np.tanh(self.s * x)

class Softplus:
    def __init__(self, s=1):
        self.s = s

    def __call__(self, x):
        return np.log(1 + np.exp(self.s * x))
    
class Identity:
    def __call__(self, x):
        return x


def make_centers(n_vars_out, n_vars_in, n_basis, centers, X=None):
    """
    Make the centers for the basis functions.

    Parameters:
    - n_vars_out: Number of output variables.
    - n_vars_in: Number of input variables.
    - n_basis: Number of basis functions.
    - centers: Method to determine the centers of the basis functions.
    - X: Input data.

    Returns:
    - The centers of the basis functions. ndarray of shape (n_vars_out, n_vars_in, n_basis)
    """
    if centers == "random":
        return np.random.uniform(0, 1, (n_vars_out, n_vars_in, n_basis))
    elif centers == "equally_spaced":
        return np.tile(np.linspace(0, 1, n_basis), (n_vars_out, n_vars_in, 1))
    elif centers == "random_data_points":
        C = np.empty((n_vars_out, n_vars_in, n_basis))
        for q in range(n_vars_out):
            for p in range(n_vars_in):
                C[q, p, :] = np.random.choice(X[:, p], n_basis)
        return C
    else:
        raise ValueError(
            "Possible values for 'centers' are 'random', 'equally_spaced', or 'random_data_points'."
        )


def apply_basis_fn(X, centers_arr, basis_fn, q, p):
    """
    Apply the basis function to the difference between the input data and the centers.

    Parameters:
    - X: Input data. Column vector of shape (n_samples, n_vars_in).
    - centers_arr: Centers of the basis functions. ndarray of shape (n_vars_out, n_vars_in, n_basis).
    - basis_fn: Basis function to apply.
    - q: Index of the output variable.
    - p: Index of the input variable.

    Returns:
    - The result of applying the basis function to the difference between the input data and the centers.
    """
    n_samples, n_vars_in = X.shape
    n_vars_out, _, n_basis = centers_arr.shape
    return basis_fn(
        X[:, p].reshape(n_samples, 1) - centers_arr[q, p, :].reshape(1, n_basis)
    )


class ExpandingLayer(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        n_vars_out,
        n_basis=10,
        centers="random",
        basis_fn=Sigmoid(),
        base_regressor=None,
        batch_size=None,
        n_jobs=1,
        output_dtype=np.float32,
        use_gpu=True,
        gpu_chunk_size=None,
    ):
        """
        Initialize the ExpandingLayer.

        Parameters:
        - n_vars_out: Number of output variables.
        - n_basis: Number of basis functions.
        - centers: Method to determine the centers of the basis functions.
        - basis_fn: Basis function to use, default is sigmoid.
        - base_regressor: Base regressor to use, default is LinearRegression.
        - batch_size: Number of samples per batch when transforming (None -> full batch).
        - n_jobs: Parallel workers for transform (1 -> sequential).
        - output_dtype: dtype to use for the transformed output buffer.
        """
        self.n_vars_out = n_vars_out
        self.n_basis = n_basis
        self.centers = centers
        self.basis_fn = basis_fn
        self.base_regressor = base_regressor
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.output_dtype = np.dtype(output_dtype)
        self.use_gpu = use_gpu
        self.gpu_chunk_size = gpu_chunk_size

    def fit(self, X, y=None):
        """
        Fit the model using the input data X and target y.

        Parameters:
        - X: Input data.
        - y: Target data.

        Returns:
        - self: Fitted estimator.
        """

        if self.base_regressor is None:
            self.base_regressor = LinearRegression(fit_intercept=False)

        self.n_samples_, self.n_vars_in_ = X.shape

        self.centers_arr_ = make_centers(
            self.n_vars_out, self.n_vars_in_, self.n_basis, self.centers, X
        )
        assert self.centers_arr_.shape == (
            self.n_vars_out,
            self.n_vars_in_,
            self.n_basis,
        ), (
            f"Centers shape is {self.centers_arr_.shape}, "
            f"expected {(self.n_vars_out, self.n_vars_in_, self.n_basis)}"
        )

        self.models_ = []

        def _fit_one(q, p):
            feats = apply_basis_fn(X, self.centers_arr_, self.basis_fn, q, p)
            reg_local = clone(self.base_regressor).fit(feats, y)
            return q, p, reg_local

        qp_iter = list(itertools.product(range(self.n_vars_out), range(self.n_vars_in_)))

        if self.n_jobs and self.n_jobs != 1:
            # Parallelise over (q,p) pairs with progress tracking.
            with tqdm_joblib(
                tqdm.tqdm(
                    total=len(qp_iter),
                    desc="Fitting 1D regressors",
                    mininterval=MININTERVAL,
                    disable=TQDM_DISABLE,
                )
            ) as progress_bar:
                # TODO: Using n_jobs = -1 can lead to memory errors. 
                # Adjust according to a fraction of remaining cores instead
                results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                    delayed(_fit_one)(q, p) for (q, p) in qp_iter
                )
            self.models_.extend(results)
        else:
            fit_iter_ = qp_iter
            if not TQDM_DISABLE:
                fit_iter_ = tqdm.tqdm(
                    qp_iter,
                    desc="Fitting 1D regressors",
                    total=len(qp_iter),
                    mininterval=MININTERVAL,
                    disable=TQDM_DISABLE,
                )
            for q, p in fit_iter_:
                self.models_.append(_fit_one(q, p))

        return self

    def transform(self, X, mode='GPU'):
        """
        Transform the input data X.

        Parameters:
        - X: Input data to transform.

        Returns:
        - Transformed data.
        """
        assert (
            self.n_vars_in_ == X.shape[1]
        ), f"Input data has {X.shape[1]} features but expected {self.n_vars_in_}"

        # Optional GPU-accelerated path with CuPy
        if self.use_gpu and CUPY_OK and mode == 'GPU':
            return self._transform_gpu(X)

        n_samples = X.shape[0]
        if n_samples == 0:
            return np.empty(
                (self.n_vars_out, self.n_vars_in_, 0), dtype=self.output_dtype
            )

        batch_size = self.batch_size or n_samples
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        batches = []

        def _predict_for_pair(batch_X, q, p, reg):
            transformed = apply_basis_fn(batch_X, self.centers_arr_, self.basis_fn, q, p)
            preds = reg.predict(transformed)
            return q, p, np.asarray(preds, dtype=self.output_dtype)

        starts = list(range(0, n_samples, batch_size))

        def _process_one(start_idx):
            stop = min(start_idx + batch_size, n_samples)
            batch_X = X[start_idx:stop]
            batch_len = stop - start_idx
            # compute all (q,p) predictions sequentially within this batch
            results_local = [
                _predict_for_pair(batch_X, q, p, reg) for q, p, reg in self.models_
            ]
            batch_out_local = np.empty(
                (self.n_vars_out, self.n_vars_in_, batch_len), dtype=self.output_dtype
            )
            for q, p, preds in results_local:
                batch_out_local[q, p, :] = preds.reshape(batch_len)
            return batch_out_local

        if self.n_jobs and self.n_jobs != 1:
            # with tqdm_joblib(tqdm.tqdm(desc="Parallelized input transformation", total=len(starts))) as progress_bar:
            batches = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(_process_one)(start) for start in starts
            )
        else:
            batches = [_process_one(start) for start in tqdm.tqdm(starts)]

        return np.concatenate(batches, axis=2) if len(batches) > 1 else batches[0]

    def _transform_gpu(self, X):
        """
        - Eliminates the inner per‑(q,p) Python loop and replaces P GEMVs with a single batched contraction per q:
            Φ_q = ψη(X[:, :, None] − C_q[None, :, :]) ∈ ℝ^{N×P×B}
            preds_q = einsum('npb,pb->np', Φ_q, W_q) + b_q
        - Fewer GPU kernel launches and much less Python overhead.
        - Head weights (coef_, intercept_) are transferred once and reused.
        Notes:
            - This function is self‑contained: it builds and caches device weights on the first call and handles rare regressors without coef_ via a fallback branch.
            - Keep dtype at float32 on device; if you need more speed and can tolerate small numeric differences, test float32/bfloat32 for Φ and W with float32 accumulations.

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        n_samples = X.shape[0]
        if n_samples == 0:
            return np.empty((self.n_vars_out, self.n_vars_in_, 0), dtype=self.output_dtype)
            
        # Cache GPU weights once (coef_, intercept_) to avoid per-call host→device copies
        if not hasattr(self, "_W_cp") or self._W_cp is None:
            # W: [Q, P, B], b: [Q, P]
            W_host = np.zeros((self.n_vars_out, self.n_vars_in_, self.n_basis), dtype=np.float32)
            b_host = np.zeros((self.n_vars_out, self.n_vars_in_), dtype=np.float32)
            # Track which pairs lack coef_ (rare; fallback if needed)
            fallback_mask = np.zeros((self.n_vars_out, self.n_vars_in_), dtype=bool)

            for (q, p, reg) in self.models_:
                coef = getattr(reg, "coef_", None)
                if coef is None:
                    fallback_mask[q, p] = True
                    continue
                coef = np.asarray(coef).ravel().astype(np.float32, copy=False)  # [B]
                W_host[q, p, :] = coef
                b_host[q, p] = float(getattr(reg, "intercept_", 0.0))

            self._W_cp = cp.asarray(W_host)           # [Q,P,B]
            self._b_cp = cp.asarray(b_host)           # [Q,P]
            self._fallback_mask = fallback_mask       # [Q,P]

        # Preload centers and basis config to GPU
        centers_cp = cp.asarray(self.centers_arr_, dtype=cp.float32)  # [Q,P,B]
        basis_name = type(self.basis_fn).__name__.lower()
        s_val = float(getattr(self.basis_fn, "s", 1.0))

        # Chunk over samples to bound memory
        chunk = self.gpu_chunk_size or n_samples
        outputs = []
        outer_iter = range(0, n_samples, chunk)

        for start in outer_iter:
            stop = min(start + chunk, n_samples)
            batch = X[start:stop]
            batch_len = stop - start
            batch_out = np.empty((self.n_vars_out, self.n_vars_in_, batch_len), dtype=self.output_dtype)

            # Move batch to device once
            x_cp = cp.asarray(batch, dtype=cp.float32)  # [N,P]

            # For each q, compute all p at once: Φ_q ∈ [N,P,B], preds_q ∈ [N,P]
            for q in range(self.n_vars_out):
                centers_q = centers_cp[q]           # [P,B]
                Wq = self._W_cp[q]                  # [P,B]
                bq = self._b_cp[q]                  # [P]

                # Broadcast differences for all p: [N,P,B]
                diff = x_cp[:, :, None] - centers_q[None, :, :]
                Phi = self.gpu_basis(diff, basis_name, s_val)               # [N,P,B]

                # Main fast path: einsum over B to get [N,P]
                preds_q = cp.einsum('npb,pb->np', Phi, Wq) + bq[None, :]  # [N,P]

                # If any (q,p) lacked coef_, recompute those columns via CPU fallback
                # (rare; keeps code robust to non-linear base regressors)
                if np.any(self._fallback_mask[q]):
                    # For those p, run reg.predict on CPU from Phi[:,p,:]
                    for p in np.where(self._fallback_mask[q])[0]:
                        reg = next(reg for (qq, pp, reg) in self.models_ if qq == q and pp == p)
                        # Copy only the needed slice back to host for predict
                        phi_np = cp.asnumpy(Phi[:, p, :])
                        preds_np = reg.predict(phi_np).astype(self.output_dtype, copy=False)
                        preds_q[:, p] = cp.asarray(preds_np, dtype=cp.float32)

                # Assign to batch_out with transpose to shape [P,N] → [N] per index
                batch_out[q, :, :] = cp.asnumpy(preds_q).T.astype(self.output_dtype, copy=False)

            outputs.append(batch_out)
            # Free per-batch GPU allocations
            del x_cp, diff, Phi, preds_q, batch_out
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()

        return np.concatenate(outputs, axis=2) if len(outputs) > 1 else outputs[0]

    @staticmethod
    def gpu_basis(diff, basis_name, s_val):
        if basis_name == "sigmoid":
            return 1.0 / (1.0 + cp.exp(-s_val * diff))
        if basis_name == "gaussian":
            return cp.exp(-((s_val * diff) ** 2))
        if basis_name == "relu":
            return cp.maximum(0.0, s_val * diff)
        if basis_name == "tanh":
            return cp.tanh(s_val * diff)
        if basis_name == "softplus":
            return cp.log1p(cp.exp(s_val * diff))
        return diff  # identity

class ConnectingLayer(TransformerMixin, RegressorMixin, BaseEstimator):

    def __init__(self, base_regressor=None):
        self.base_regressor = base_regressor

    def fit(self, X, y=None):
        if self.base_regressor is None:
            self.base_regressor = LinearRegression(fit_intercept=True)

        self.n_vars_out_, self.n_vars_in_, self.n_samples_ = X.shape
        self.models_ = []

        for q in tqdm.tqdm(
            range(self.n_vars_out_),
            desc="Fitting connecting regressors",
            total=self.n_vars_out_,
            mininterval=MININTERVAL,
            disable=TQDM_DISABLE,
        ):
            reg = clone(self.base_regressor).fit(X[q, :, :].T, y)
            self.models_.append((q, reg))

        return self

    def transform(self, X):
        assert (
            self.n_vars_out_ > 1
        ), "Unable to transform data with only one output variable. Use predict instead."

        out = np.empty((self.n_vars_out_, X.shape[2]))

        for q, reg in self.models_:
            out[q, :] = reg.predict(X[q, :, :].T)

        return out.T

    def predict(self, X):
        assert (
            self.n_vars_out_ == 1
        ), "Unable to predict data with more than one output variable. Use transform instead."

        _, reg = self.models_[0]
        return reg.predict(X[0, :, :].T)


def make_hkan_layer(
    *,
    layer_idx,
    n_vars_out,
    n_basis=10,
    centers="random",
    basis_fn=Sigmoid(),
    expanding_base_regressor=None,
    connecting_base_regressor=None,
    batch_size=128,
    n_jobs=2,
    use_gpu=True,
    gpu_chunk_size=None
):
    """Pipeline of ExpandingLayer and ConnectingLayer."""
    steps = [
        (
            f"expanding_layer_{layer_idx}",
            ExpandingLayer(
                n_vars_out=n_vars_out,
                n_basis=n_basis,
                centers=centers,
                basis_fn=basis_fn,
                base_regressor=expanding_base_regressor,
                batch_size=batch_size,
                n_jobs=n_jobs,
                use_gpu=use_gpu,
                gpu_chunk_size=gpu_chunk_size
            ),
        ),
        (
            f"connecting_layer_{layer_idx}",
            ConnectingLayer(base_regressor=connecting_base_regressor),
        ),
    ]
    return Pipeline(steps)


def extend_hkan(
    model,
    *,
    layer_idx=None,
    n_vars_out=1,
    n_basis=10,
    centers="random",
    basis_fn=Sigmoid(),
    expanding_base_regressor=None,
    connecting_base_regressor=None,
):
    """
    Extend the HKAN model with additional HKAN layer.

    Parameters:
    - model: The HKAN model to extend.
    - n_vars_out: Number of output variables.
    - n_basis: Number of basis functions.
    - centers: Method to determine the centers of the basis functions.
    - basis_fn: Basis function to use, default is sigmoid.
    - expanding_base_regressor: Base regressor for the expanding layer.
    - connecting_base_regressor: Base regressor for the connecting layer.

    Returns:
    - The extended HKAN model.
    """
    if layer_idx is None:
        layer_idx = len(model.steps) // 2

    new_layer = make_hkan_layer(
        layer_idx=layer_idx,
        n_vars_out=n_vars_out,
        n_basis=n_basis,
        centers=centers,
        basis_fn=basis_fn,
        expanding_base_regressor=expanding_base_regressor,
        connecting_base_regressor=connecting_base_regressor,
    )

    return Pipeline(model.steps + new_layer.steps)

