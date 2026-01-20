import numpy as np
from scipy.stats import multivariate_normal

def AR1_model(alpha, eps):
    """
    AR1 (decreasing correlation) model for two-sample testing:
    - X ~ N(0,\Sigma_1), where \Sigma_1(i,j) = alpha^|i-j|
    - Y ~ N(0,\Sigma_2), where \Sigma_2(i,j) = (alpha + eps)^|i-j|

    Parameters:
    alpha (float): Base correlation coefficient for the covariance matrix of X.
    eps (float): Additional correlation coefficient for the covariance matrix of Y.

    Returns:
    themodel (class): A model class with sampling methods.
    
    Class Attributes:
    __name__ (str): The name of the model.
    params (dict): A dictionary containing the parameters 'eps' and 'P'.
    
    Methods:
    sample_X(n): Samples n points from the distribution X.
    sample_Y(n): Samples n points from the distribution Y.
    """
    class themodel:
        __name__ = ',Gaussian, Decreasing Correlation'
        def __init__(self, d, alpha =alpha, eps=eps):
            self.params = {'alpha' : alpha, 'eps': eps}
            self.d = d

        def sample_X(self, n):
            cov_X = np.array([[alpha ** abs(i - j) for j in range(self.d)] for i in range(self.d)])
            return np.random.multivariate_normal(mean=np.zeros(self.d), cov=cov_X, size=n)
        def sample_Y(self, n):
            cov_Y = np.array([[ (alpha + eps) ** abs(i - j) for j in range(self.d)] for i in range(self.d)])
            return np.random.multivariate_normal(mean=np.zeros(self.d), cov=cov_Y, size=n)
    return themodel