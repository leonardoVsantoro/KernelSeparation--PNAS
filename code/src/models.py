import numpy as np


def AR1_model(alpha, eps):
    """
    AR1 (decreasing correlation) model for two-sample testing:
      X ~ N(0, Sigma_1), where Sigma_1[i,j] = alpha^|i-j|
      Y ~ N(0, Sigma_2), where Sigma_2[i,j] = (alpha + eps)^|i-j|

    Parameters
    ----------
    alpha : float
        Base correlation coefficient for the covariance matrix of X.
    eps : float
        Additional correlation offset for the covariance matrix of Y.

    Returns
    -------
    class
        Model class with sampling methods sample_X(n) and sample_Y(n).
    """
    class _Model:
        __name__ = "Gaussian_DecreasingCorrelation"

        def __init__(self, d):
            self.d = d
            self.params = {"alpha": alpha, "eps": eps}

        def sample_X(self, n):
            cov = np.array([[alpha ** abs(i - j) for j in range(self.d)]
                            for i in range(self.d)])
            return np.random.multivariate_normal(np.zeros(self.d), cov, size=n)

        def sample_Y(self, n):
            cov = np.array([[(alpha + eps) ** abs(i - j) for j in range(self.d)]
                            for i in range(self.d)])
            return np.random.multivariate_normal(np.zeros(self.d), cov, size=n)

    return _Model
