import numpy as np
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    ElasticNet,
    Lars,
    BayesianRidge,
)

from hkan import Identity, Tanh, extend_hkan, make_hkan_layer


LINEAR_HEAD_REGISTRY = {
    "linear": lambda alpha: LinearRegression(),
    "ridge": lambda alpha: Ridge(alpha=alpha),
    "elasticnet": lambda alpha: ElasticNet(alpha=alpha, l1_ratio=0.5),
    "lars": lambda alpha: Lars(),
    "bayesridge": lambda alpha: BayesianRidge(),
}


def get_linear_factory(kind: str, alpha: float):
    key = kind.lower()
    if key not in LINEAR_HEAD_REGISTRY:
        raise ValueError(f"Unsupported linear head '{kind}'. Options: {list(LINEAR_HEAD_REGISTRY)}")
    return lambda: LINEAR_HEAD_REGISTRY[key](alpha)


def build_hkan(
    n_vars_out=32,
    n_basis=23,
    infer_batch_size=128,
    n_jobs=2,
    use_gpu=True,
    gpu_chunk_size=None,
    basis_fn=Tanh(s=50)
):
    model = make_hkan_layer(
        layer_idx=0,
        n_vars_out=n_vars_out,
        basis_fn=basis_fn,
        n_basis=n_basis,
        centers="random_data_points",  # "random",
        expanding_base_regressor=Ridge(alpha=0.01),
        batch_size=infer_batch_size,
        n_jobs=n_jobs,
        use_gpu=use_gpu,
        gpu_chunk_size=gpu_chunk_size,
    )

    model = extend_hkan(
        model,
        layer_idx=1,
        n_vars_out=1,
        basis_fn=Identity(),
        centers="random_data_points",
        expanding_base_regressor=Ridge(alpha=0.1),
    )
    return model


def get_default_masks():
    mask_A = np.array([0, 1])
    mask_B = np.array([2, 3, 4])
    mask_C = np.array([5, 6, 7])
    mask_D = np.array([8, 9])
    return mask_A, mask_B, mask_C, mask_D
