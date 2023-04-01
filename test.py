import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

z = np.random.multivariate_normal(mean=np.array([0, 0]), cov=np.array([[1, 0.4], [0.4, 1]]), size=1000)
y = np.transpose(z)
plt.scatter(y[0], y[1])

plt.show()
