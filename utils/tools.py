import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import pandas as pd 

def inv_sqrtm_ED(eigdec):
    '''
    Inverse square-root of a psd matrix given its eigendecomposition
    -----------------------------------------------
    Inputs:
        eigdec: tuple (eigenvalues, eigenvectors)
    Outputs:
        inv_sqrtm_matrix: inverse of the matrix
    -----------------------------------------------
    '''
    S, U = eigdec
    return U @ np.diag(np.array([s**-0.5 if s > 0 else 0 for s in S])) @ U.T

def sqrtm_ED(eigdec):
    '''
    Square-root of a psd matrix given its eigendecomposition
    -----------------------------------------------
    Inputs:
        eigdec: tuple (eigenvalues, eigenvectors)
    Outputs:
        sqrtm_matrix: inverse of the matrix
    -----------------------------------------------
    '''
    S, U = eigdec
    return U @ np.diag(np.array([s**0.5 if s > 0 else 0 for s in S])) @ U.T

def inv_ED(eigdec):
    '''
    -----------------------------------------------
    Inverse of a psd matrix given its eigendecomposition
    Inputs:
        eigdec: tuple (eigenvalues, eigenvectors)
    Outputs:
        inv_matrix: inverse of the matrix
    -----------------------------------------------
    '''
    S, U = eigdec
    return U @ np.diag(np.array([1/s if s > 0 else 0 for s in S])) @ U.T


def MC_test_vals(model, Ns, K, kernel_type):
    '''
    Compute MC test values for KLR and MMD under null and alternative hypotheses
    -----------------------------------------------
    Inputs:
        Ns: list of sample sizes
        K: number of MC repetitions
        model: data model with methods sample_X(N) and sample_Y(N)
        ridge: regularization parameter for KLR
        kernel_type: type of kernel for distance computation
    Outputs:
        df: pandas DataFrame with columns ["Hypothesis", "Value", "Statistic", "Sample Size"]
    -----------------------------------------------
    '''
    dfs = []
    for N in Ns:
        ridge =N**(-1/5)
        MMD_null,KLR_null,MMD_alt,KLR_alt = np.zeros(K),np.zeros(K),np.zeros(K),np.zeros(K)
        for k in tqdm(range(K), total = K):
            KLR_null[k], MMD_null[k] = compute_stats(model.sample_X(N), model.sample_X(N), ridge = ridge, kernel_type = kernel_type)      # ----- null
            KLR_alt[k], MMD_alt[k] = compute_stats(model.sample_X(N), model.sample_Y(N), ridge = ridge, kernel_type = kernel_type)        # ----- alternative 
        data = (
            [("Null", x, "MMD", N) for x in MMD_null] +
            [("Alternative", x, "MMD", N) for x in MMD_alt] +
            [("Null", x, "KLR", N) for x in KLR_null] +
            [("Alternative", x, "KLR", N) for x in KLR_alt]
        )
        dfs.append(pd.DataFrame(data, columns=["Hypothesis", "Value", "Statistic", "Sample Size"]))
    return pd.concat(dfs, ignore_index=True)


def compute_stats(X, Y, ridge = None, kernel_type = 'sqeuclidean'):
    """
    Compute (KLR, MMD) for samples X, Y
    -----------------------------------------------
    Inputs:
        X: np.array of shape (N, d)
        Y: np.array of shape (N, d)
        ridge: regularization parameter for KLR
        kernel_type: type of kernel for distance computation
    Outputs:
        KLR_val: KLR statistic value
        MMD_val: MMD statistic value
    -----------------------------------------------
    """
    N = X.shape[0]
    if ridge is None:
        ridge =N**(-1/5)
    
    # ---------- kernel ----------
    fullsample = np.concatenate([X, Y]).reshape(2 * N, -1)
    pairwise_dists = cdist(fullsample, fullsample, kernel_type)
    bandwidth = 2.0 * np.median(pairwise_dists[pairwise_dists > 0])
    kernel_matrix = np.exp(-pairwise_dists / bandwidth)

    # ---------- KLR ----------
    eigvals, eigvecs = np.linalg.eigh(kernel_matrix + 1e-5 * np.eye(2 * N))
    K_inv_sqrt = inv_sqrtm_ED((eigvals, eigvecs))
    kxx = kernel_matrix [:N, :N]
    kyy = kernel_matrix [N:, N:]
    kxy = kernel_matrix [:N, N:]
    kyx = kxy.T

    KX = K_inv_sqrt @ np.concatenate([kxx, kyx]) # adjust for non-orthonormality of kernel features
    KY = K_inv_sqrt @ np.concatenate([kxy, kyy]) # -- '' --

    SX = (KX @ KX.T) / N; SY = (KY @ KY.T) / N
    S_avg = .9*SX + .1*SY

    eigvals_S, eigvecs_S = np.linalg.eigh(S_avg + ridge * np.eye(2 * N))
    S_inv_sqrt = inv_sqrtm_ED((eigvals_S, eigvecs_S))
    lam = np.linalg.eigvalsh(S_inv_sqrt @ (S_avg - SY) @ S_inv_sqrt)
    # lam = np.clip(lam, -1 + 1e-12, None)
    # KLR_val = 0.5 * np.abs( -np.sum(lam) + np.sum(np.log1p(lam)) )
    KLR_val = np.sum(lam**2) # simple approximation

    # ---------- MMD ----------
    MMD_val = np.mean(kxx) + np.mean(kyy) - 2 * np.mean(kxy)


    return KLR_val, MMD_val


