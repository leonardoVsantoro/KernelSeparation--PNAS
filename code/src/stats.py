import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import pandas as pd


def inv_sqrtm_ED(eigdec):
    """Inverse square-root of a PSD matrix from its eigendecomposition."""
    S, U = eigdec
    return U @ np.diag([s ** -0.5 if s > 0 else 0.0 for s in S]) @ U.T


def sqrtm_ED(eigdec):
    """Square-root of a PSD matrix from its eigendecomposition."""
    S, U = eigdec
    return U @ np.diag([s ** 0.5 if s > 0 else 0.0 for s in S]) @ U.T


def inv_ED(eigdec):
    """Inverse of a PSD matrix from its eigendecomposition."""
    S, U = eigdec
    return U @ np.diag([1.0 / s if s > 0 else 0.0 for s in S]) @ U.T


def compute_stats(X, Y, ridge=None, kernel_type="sqeuclidean"):
    """
    Compute (KLR, MMD) test statistics for samples X and Y.

    Parameters
    ----------
    X, Y : np.ndarray, shape (N, d)
    ridge : float, optional
        Regularization parameter; defaults to N^{-1/5}.
    kernel_type : str
        Distance metric passed to scipy.spatial.distance.cdist.

    Returns
    -------
    KLR_val, MMD_val : float
    """
    N = X.shape[0]
    if ridge is None:
        ridge = N ** (-1 / 5)

    # Kernel matrix
    fullsample = np.concatenate([X, Y]).reshape(2 * N, -1)
    pairwise_dists = cdist(fullsample, fullsample, kernel_type)
    bandwidth = 2.0 * np.median(pairwise_dists[pairwise_dists > 0])
    K = np.exp(-pairwise_dists / bandwidth)

    # KLR statistic
    eigvals, eigvecs = np.linalg.eigh(K + 1e-5 * np.eye(2 * N))
    K_inv_sqrt = inv_sqrtm_ED((eigvals, eigvecs))

    kxx, kyy, kxy = K[:N, :N], K[N:, N:], K[:N, N:]
    kyx = kxy.T

    KX = K_inv_sqrt @ np.concatenate([kxx, kyx])
    KY = K_inv_sqrt @ np.concatenate([kxy, kyy])

    SX = (KX @ KX.T) / N
    SY = (KY @ KY.T) / N
    S_avg = 0.9 * SX + 0.1 * SY

    eigvals_S, eigvecs_S = np.linalg.eigh(S_avg + ridge * np.eye(2 * N))
    S_inv_sqrt = inv_sqrtm_ED((eigvals_S, eigvecs_S))
    lam = np.linalg.eigvalsh(S_inv_sqrt @ (S_avg - SY) @ S_inv_sqrt)
    KLR_val = np.sum(lam ** 2)

    # MMD statistic
    MMD_val = np.mean(kxx) + np.mean(kyy) - 2 * np.mean(kxy)

    return KLR_val, MMD_val


def MC_test_vals(model, Ns, K, kernel_type):
    """
    Run Monte Carlo experiments for KLR and MMD under null and alternative.

    Parameters
    ----------
    model : object with sample_X(n) and sample_Y(n) methods
    Ns : list of int
        Sample sizes to evaluate.
    K : int
        Number of Monte Carlo repetitions.
    kernel_type : str
        Kernel/distance type passed to compute_stats.

    Returns
    -------
    pd.DataFrame
        Columns: ["Hypothesis", "Value", "Statistic", "Sample Size"]
    """
    dfs = []
    for N in Ns:
        ridge = N ** (-1 / 5)
        MMD_null = np.zeros(K); KLR_null = np.zeros(K)
        MMD_alt  = np.zeros(K); KLR_alt  = np.zeros(K)
        for k in tqdm(range(K), total=K):
            KLR_null[k], MMD_null[k] = compute_stats(
                model.sample_X(N), model.sample_X(N),
                ridge=ridge, kernel_type=kernel_type)
            KLR_alt[k], MMD_alt[k] = compute_stats(
                model.sample_X(N), model.sample_Y(N),
                ridge=ridge, kernel_type=kernel_type)
        rows = (
            [("Null",        x, "MMD", N) for x in MMD_null] +
            [("Alternative", x, "MMD", N) for x in MMD_alt]  +
            [("Null",        x, "KLR", N) for x in KLR_null] +
            [("Alternative", x, "KLR", N) for x in KLR_alt]
        )
        dfs.append(pd.DataFrame(rows, columns=["Hypothesis", "Value", "Statistic", "Sample Size"]))
    return pd.concat(dfs, ignore_index=True)
