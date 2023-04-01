"""GaussianMultivariate module."""
import logging
import numpy as np
from scipy import stats
from utils import pseudo_inv_ecdf
from statsmodels.distributions.empirical_distribution import ECDF
LOGGER = logging.getLogger(__name__)


class MetaGaussianMultivariate:
    """Class for a multivariate distribution that uses the Gaussian copula.
    Args:
        distribution (str or dict):
            Fully qualified name of the class to be used for modeling the marginal
            distributions or a dictionary mapping column names to the fully qualified
            distribution names.
    """
    marginals = None
    inv_marginals = None
    cov = None
    mu = None
    pval = None
    fitted = False

    def fit(self, X):
        """Compute the distribution for each variable and then its covariance matrix.
        Arguments:
            X (numpy.array):
                Values of the random variables.
        """
        assert isinstance(X, np.ndarray)
        d = X.shape[1]
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
        mu = np.mean(X, axis=0)

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
        if not self.fitted:
            raise ValueError('The disribution is not fitted yet. First provide a sample of n elements X (nxd) and fit the model with MetaGaussianMultivariate.fit(X)')
        return

    def probability_density(self, X):
        """Compute the probability density for each point in X.
        Arguments:
            X (pandas.DataFrame):
                Values for which the probability density will be computed.
        Returns:
            numpy.ndarray:
                Probability density values for points in X.
        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        if self.fitted:
            pass
        else:
            raise ValueError('The disribution is not fitted yet. First provide a sample of n elements X (nxd) and fit the model with MetaGaussianMultivariate.fit(X)')

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution value for each point in X.
        Arguments:
            X (pandas.DataFrame):
                Values for which the cumulative distribution will be computed.
        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.
        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        if self.fitted:
            pass
        else:
            raise ValueError()

    def calc_inv_marginals(self):
        inv_marginals = [lambda x: pseudo_inv_ecdf(F, x) for F in self.marginals]
        self.inv_marginals = inv_marginals
        return

    def simulate(self, n):
        """Sample values from this model.
        Argument:
            num_rows (int):
                Number of rows to sample.
            conditions (dict or pd.Series):
                Mapping of the column names and column values to condition on.
        Returns:
            numpy.ndarray:
                Array of shape (n_samples, *) with values randomly
                sampled from this model distribution. If conditions have been
                given, the output array also contains the corresponding columns
                populated with the given values.
        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        if not self.inv_marginals:
            self.calc_inv_marginals()

        X = np.random.multivariate_normal(self.mu, self.cov, size=n)
        U = stats.norm.cdf(X)
        Y = np.array([[F_inv(U[j, i]) for i, F_inv in enumerate(self.inv_marginals)] for j in range(U.shape[0])])
        return Y
