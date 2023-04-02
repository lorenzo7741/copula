# Copula Repository
Repo with multivariate copula classes and calibration methods
The Repo contains:
- A class MetaGaussianMultivariate: Class for a multivariate meta-gaussian copula random variable. The class provide functions to fit and 
    simulate these kind of copulas
- fx_rates_application: an application of the MetaGaussianMultivariate class to the pair of 
exchange rates $/£ and £/€.
## An application of Multivariate Meta-Gaussian Copula Model for Log Returns of exchange rate USD/GBP and USD/EUR
This is an example of how a meta-Gaussian copula can be fitted to a pair of prices that shows
a correlation and a fat-tails effect. This specific example was build on two 
historical series: the exchange rate $/£ and the exchange rate $/€. The historical series 
are provided by FRED Federal Reserve Economic Data (FRED), that is a database maintained by 
the Research division of the Federal Reserve Bank. The code is all available in fx_rates_application.py
Let's see the procedure:

These are the prices:
![alt text](https://github.com/lorenzo7741/copula/blob/pp/plot/rates_hist.png?raw=true)
This is a scatter plot of the log returns and a compareson between each ECDF and its
 respective Gaussian normal cumulative density function with the mean and standar
  deviation of the respective sample.
![alt text](https://github.com/lorenzo7741/copula/blob/pp/plot/rates_hist_scatter.png?raw=true)
![alt text](https://github.com/lorenzo7741/copula/blob/pp/plot/ecdf_vs_gaussian.png?raw=true)
Prices show a certain correlation. Moreover a little
fat-tails effect is showd by the ECDFs. A meta-Gaussian model, can be used to model the 
joint-cumulative density function of this pair. Tha class MetaGaussianMultivariate is used, 
with its main method .fit on the returns of the prices. 

Once the model is fitted, and the distribution function is defined, a sample
of $n=1000$ is generated. The following chart shows the scatter plot of a sample of $n$ 
following the fitted meta-Gaussian distribution
![alt text](https://github.com/lorenzo7741/copula/blob/pp/plot/sim_copula_scatter.png?raw=true)

Now we can compare the meta-Gaussian methodology with the classical Gaussian model. We therefore 
evaluate the mean and the covariance matrix of the historical sample of the log-returns. 
The resulting model, once simulated, has got this scatter plot 
![alt text](https://github.com/lorenzo7741/copula/blob/pp/plot/sim_gaussian_scatter.png?raw=true)


## Copula's Theory and Main Results

In probability theory, a copula is a mathematical object used to 
describe the dependence between two or more random variables, without taking their 
marginal distributions into account. In other words, a copula represents the joint 
distribution of a set of random variables as a uniform distribution over the unit 
square/cube, factoring in their dependence structure separate from their marginal 
distributions.

### Definition (Copula)
A $d$-dimensional copula, $C : [0, 1]^d : → [0, 1]$ is a cumulative distribution 
function (CDF) of uniform marginals. Therefore $C$ is a copula if there exists a random vector 
$X = (X_1, X_2, ..., X_d)$ such that each $X_i$ is uniform in $[0, 1]$ and such that
$C(x_1,...,x_d) = P(X_1\leq x_1,..., X_d\leq x_d)$

The main result about copula is the Sklar's theorem. Sklar's Theorem is a fundamental 
theorem in the theory of copulas and provides a way to decompose any multivariate
 distribution function into marginals and a copula.

### Sklar's Theorem
For any multivariate distribution function $F(x₁, x₂, ..., xn)$ of a random variable 
$X = (X_1, X_2, ..., X_d)$ with arbitrary marginals, there exists a copula $C(u_1, ..., u_n)$ such that
```math
F(x_1, x_2, \cdots, x_n) = C(F_1(x_1), F_2(x_2), \cdots, F_n(x_n))
```
where $F_i$ are the marginal distribution functions of $X_1, X_2, ..., X_d$

### Gaussian Copulas
The Gaussian Copula is a copula that uses a Gaussian distribution 
(normal distribution) to model the dependency structure between random variables.
The Gaussian Copula has become increasingly popular in finance as a tool to measure 
credit risk and correlation between prices. The definition of the Gaussian copula
is a simple application of the Sklar's theorem at the case of a Gaussian vector.

### Definition (Gaussian Copula)
Let $X ∼ \tilde N_d(0, P)$, where P is a correlation matrix. From Sklar's theorem we know
that there exists a copula $C$ such that
```math
\Phi(x_1, x_2, \cdots, x_n) = C(\phi(x_1), \phi(x_2), \cdots, \phi(x_n))
```
where $\phi$ is the standard univariate Gaussian cumulative density function of a 
 random variable $\tilde N(0, 1)$ and $\Phi$ denotes the joint cumulative density function
of a multivariate Gaussian distribution $\tilde N_d(0, P)$. $C$ is called Gaussian copula.


From now on we will refer to multivariate random variables with Gaussian copula
and arbitrary marginals as meta-Gaussian random variables

### Definition (Kendall's tau)
Let $(X_1, X_2)$ and $(Y_1, Y_2)$ be independent and identically distributed random vectors
with distribution function F. The Kendall's tau is a measure of correlation
between $X_1 and $X_2$ and is defined as the probability of concordance minus 
the probability of discordance, thus the Kendall's tau of $X_1$ and $X_2)$ is defined as
```math
\tau(X_1, X_2) = P{(X_1 − Y_1)(X_2 − Y_2) > 0} − P{(X_1 − Y_1)(X_2 − Y_2) < 0}.
```

### Theorem
Let $X_1,\ldots, X_d$ be meta-Gaussian multivariate vector with correlation
matrix $P$. It holds that
```math
\tau(X_i, X_j) = \frac{2}{\pi} \arcsin(P_{ij})
```
where $P_{ij}$ are the elements of $P$
This means that Kendall's tau can be used as estimateor for the matrix $P$, and it doesn't 
depend on marginals. 