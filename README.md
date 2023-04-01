# Copula Repository
Repo with multivariate copula classes and calibration methods
The Repo contains:
- A class MetaGaussianMultivariate: Class for a multivariate meta-gaussian copula random variable. The class provide functions to fit and 
    simulate these kind of copulas
- fx_rates_application: an application of the MetaGaussianMultivariate class to the pair of 
exchange rates $/£ and £/€.
## An application of Multivariate Meta-Gaussian Copula Model for Log Returns of exchange rate `$`/`£` and \$/€
This is an example of how a meta-Gaussian copula can be fitted to a pair of prices that shows
a correlation and a fat-tails effect. This specific example was build on two 
historical series: the exchange rate $/£ and the exchange rate $/€. The historical series 
are provided by FRED Federal Reserve Economic Data (FRED), that is a database maintained by 
the Research division of the Federal Reserve Bank. The code is all available in fx_rates_application.py
Let's see the procedure:

### 1. The historical series
This are the prices:
![alt text](https://github.com/lorenzo7741/copula/blob/pp/plot/rates_hist.png?raw=true)
This is a scatter plot of the log returns and a compareson between each ECDF and its
 respective Gaussian normal cumulative density function with the mean and standar
  deviation of the respective sample.
![alt text](https://github.com/lorenzo7741/copula/blob/pp/plot/rates_hist_scatter.png?raw=true)
![alt text](https://github.com/lorenzo7741/copula/blob/pp/plot/ecdf_vs_gaussian.png?raw=true)
A certain correlation between the returns of the prices is showed. Moreover a little
fat-tails effect is showd by the ECDFs. A meta-Gaussian model, can be used to model the 
joint-cumulative density function of this pair. Tha class MetaGaussianMultivariate is used, with
its main method .fit on the returns of the prices. 

Once the model is fitted, and the distribution function is defined, a sample
 of $n=1000$ is generated. The following chart shows its scatter plot
![alt text](https://github.com/lorenzo7741/copula/blob/pp/plot/sim_copula_scatter.png?raw=true)
How about doing this with 
![alt text](https://github.com/lorenzo7741/copula/blob/pp/plot/sim_gaussian_scatter.png?raw=true)


## Copula

In probability theory, a copula is a mathematical object used to describe the dependence between two or more random variables, without taking their marginal distributions into account. In other words, a copula represents the joint distribution of a set of random variables as a uniform distribution over the unit square/cube, factoring in their dependence structure separate from their marginal distributions.

Suppose $X = (X_1, X_2, ..., X_d)$ is a random vector with continuous marginal distributions on $[0,1]$, where $d$ is the number of variables. The Copula $C$ of $X$ is a function $C : [0,1]^d \rightarrow [0,1]$ that satisfies the following properties:

1. $C$ is a multidimensional cumulative distribution function (CDF) with uniform marginals.
2. $C$ is a dependency structure that describes the probability distribution of a set of variables.
3. $C$ is Sklar's Theorem compatible, i.e., it links the marginal distribution functions of a random vector to its joint distribution through a copula.

## Sklar's Theorem

Sklar's Theorem is a fundamental theorem in the theory of copulas and provides a way to uniquely decompose any multivariate distribution function into univariate distributions and a copula. The theorem states:

- For any multivariate distribution function H(x₁, x₂, ..., xn), there exist unique marginal distribution functions Fi and a unique copula C(u₁, u₂, ..., un) such that:

\begin{equation}
H(x_1, x_2, \cdots, x_n) = C(F_1(x_1), F_2(x_2), \cdots, F_n(x_n)), \;\;\;\;\;\;\;\;\;\; x_1 \in \mathbb{R}, \cdots, x_n \in \mathbb{R}
\end{equation}
where the marginal distribution functions Fi are uniquely defined, continuous and non-decreasing.

## Gaussian Copulae

The Gaussian Copula is a special type of copula that uses a Gaussian distribution 
(normal distribution) to model the dependency structure between random variables.
 The Gaussian Copula has become increasingly popular in finance as a tool to measure 
 credit risk and to calculate Value at Risk (VaR). A multivariate normal distribution 
 with correlation matrix C can be expressed as a Gaussian copula with covariance
  matrix $Σ, where $Σ is any matrix such that $ΣΣ^T = C$.

Each element of the correlation matrix for a Gaussian Copula is bounded between -1 and 1. The parameter ρ represents the correlation coefficient between the two variables, and is given

```math
C(u_1, u_2) = \Phi_{\rho}\Big(\Phi^{-1}(u_1),\Phi^{-1}(u_2)\Big),
```
where $u_1$ and $u_2$ are parts of the standard multivariate normal random vector with correlation matrix $C$, $\Phi(z)$ is the standard
normal cumulative distribution function, and $\Phi _\rho(u_1,u_2)$ is the bivariate normal cumulative distribution
function with correlation coefficient $\rho$.
