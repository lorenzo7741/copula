import pandas as pd
import pandas_datareader as pdr
import numpy as np
from scipy.stats import norm
from MetaGaussianMultivariate import MetaGaussianMultivariate as mgm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
lm = [-0.02, 0.02]

start = datetime(2021, 1, 1)
end = datetime(2023, 3, 1)
syms = ['DEXUSUK', 'DEXUSEU']

df = pd.DataFrame()
for sym in syms:
  ts = pdr.fred.FredReader(sym, start=start, end=end)
  df1 = ts.read()
  df = pd.concat([df, df1], axis=1)

df.dropna(inplace=True)
X = df[syms].values
ret = np.log(X[1:] / X[0:-1])

cp_mgm = mgm()
cp_mgm.fit(ret)
n_sim = 1000
Y = cp_mgm.simulate(n_sim)

plt.figure()
plt.title("Exchange Rates - Historical Sample")
plt.plot(df.index, X[:, 0])
plt.plot(df.index, X[:, 1])
plt.legend(['exchange rate $/£', 'exchange rate $/€'])
plt.show()

plt.figure()
plt.title("Exchange Rates log-returns - Scatter Plot")
plt.scatter(ret[:, 0], ret[:, 1])
plt.xlim(lm)
plt.ylim(lm)
plt.show()

# Analysis on normal distribution VS empirical ones
fig, axs = plt.subplots(1, 2)
x = np.linspace(-0.02, 0.02, 5000)
mu0, std0 = norm.fit(ret[:, 0])
mu1, std1 = norm.fit(ret[:, 1])
axs[0].set_title("CDF Normal vs CDF Empirical $/£")
axs[0].plot(x, cp_mgm.marginals[0](x), 'b-')
axs[0].plot(x, norm.cdf(x, loc=mu0, scale=std0), 'b--')
axs[1].set_title("CDF Normal vs CDF Empirical $/€")
axs[1].plot(x, cp_mgm.marginals[1](x), 'r-')
axs[1].plot(x, norm.cdf(x, loc=mu1, scale=std1), 'r--')
plt.show()

plt.figure()
plt.title('Simulated meta-gaussian variables')
plt.scatter(Y[:, 0], Y[:, 1])
plt.xlim(lm)
plt.ylim(lm)
plt.show()

# Last one
plt.figure()
Z = np.random.multivariate_normal(mean=np.array([0,0]), cov=np.cov(np.transpose(ret)), size=1000)
Y = np.transpose(Z)
plt.title('Simulated Gaussian variables')
plt.scatter(Y[0], Y[1])
plt.xlim(lm)
plt.ylim(lm)
plt.show()
