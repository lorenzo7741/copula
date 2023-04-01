"""GaussianMultivariate module."""
import logging
import numpy as np
from scipy import stats
from utils import pseudo_inv_ecdf
from statsmodels.distributions.empirical_distribution import ECDF
LOGGER = logging.getLogger(__name__)


class MetaGaussianMultivariate:
    """Class for a multivariate meta-gaussian copula random variable. The class provide functions to fit and
    simulate these kind of copulas
    """
    marginals = None
    inv_marginals = None
    cov = None
    mu = None
    pval = None
    fitted = False

    def fit(self, X):
        """
        This function fits a multivariate meta-gaussian copula on the data provided in X. Marginals are supposed with arbitrary
        distribution and are and estimated empirically. The parameters of the gaussian copula are estimated with
        Kendall's tau.
        :param
             X (nxd numpy.array):  Sample data used to fit the model. `n` and `d` denote the number of observations and
             the number of variables, respectively.
        :return
            self.marginals (dx1 lambda functions):
            self.cov (dxd numpy.array): covariance matrix estimated
            self.mu (dx1 numpy.array): mean estimated
            self.pval (dx1 numpy.array): p-value of the estimation of the covariance with Kendall's tau
            self.fitted (bool): put as true from the function
        """
        assert isinstance(X, np.ndarray)
        d = X.shape[1]
        mu = np.mean(X, axis=0)
        X = X - mu
        marginals = []
        cov = np.eye(d)
        pval = np.empty((d, d))
        pval.fill(np.nan)

        for i in range(d):
            for j in range(i + 1, d):
                tau, p = stats.kendalltau(X[:, i], X[:, j])
                if np.isnan(tau):
                    raise ValueError('Unable to compute Kendall tau.')
                omega = np.sin(np.pi * tau / 2)
                cov[i, j], cov[j, i] = omega, omega
                pval[i, j], pval[j, i] = p, p

        for i in range(d):
            # LOGGER.debug('Fitting column %s to ??', column_name)
            column = X[:, i]
            ecdf = ECDF(column)
            marginals.append(ecdf)

        self.marginals = marginals
        self.cov = cov
        self.mu = mu
        self.pval = pval
        self.fitted = True
        LOGGER.debug('Meta-Gaussian Multivariate Copula fitted successfully.')

    def check_fit(self):
        """
        This function checks if the model is fitted or not. It raises a `ValueError` if the model is not fitted yet.
        :return:
            None
        """
        if not self.fitted:
            raise ValueError('The disribution is not fitted yet. First provide a sample of n elements X (nxd) and fit the model with MetaGaussianMultivariate.fit(X)')
        return

#    def probability_density(self):
#        if self.fitted:
#            pass
#        else:
#            raise ValueError('The disribution is not fitted yet. First provide a sample of n elements X (nxd) and fit the model with MetaGaussianMultivariate.fit(X)')
#    def cumulative_distribution(self):
#        if self.fitted:
#            pass
#        else:
#            raise ValueError()

    def calc_inv_marginals(self):
        """
        This function calculates the inverse of the empirical cumulative distribution function
        for marginal distributions.
        :return:
            self.inv_marginals
        """
        inv_marginals = [lambda x: pseudo_inv_ecdf(F, x) for F in self.marginals]
        self.inv_marginals = inv_marginals
        return

    def simulate(self, n):
        """
        This function samples `n` points from the fitted model and returns a `numpy.ndarray` of shape `(n,d)` where
        `d` denotes the number of variables. It raises a `ValueError` if the model is not fitted yet.
        :param
            n (int): dimension of the sample
        :returns
            (nxd numpy.ndarray): the simulated sample
        """
        self.check_fit()
        if not self.inv_marginals:
            self.calc_inv_marginals()

        X = np.random.multivariate_normal(self.mu, self.cov, size=n)
        U = stats.norm.cdf(X)
        Y = np.array([[F_inv(U[j, i]) for i, F_inv in enumerate(self.inv_marginals)] for j in range(U.shape[0])])
        return Y
