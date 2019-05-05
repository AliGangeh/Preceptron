import numpy as np
import matplotlib.pyplot as plt

npts=100
np.random.seed(0)
topreg=np.array([np.random.normal(10, 2, npts), np.random.normal(12,2,npts)]).T
botreg=np.array([np.random.normal(5,2, npts), np.random.normal(6,2,npts)]).T
_, ax = plt.subplots(figsize=(4,4))
ax.scatter(botreg[:, 0], botreg[:, 1], color="b")
ax.scatter(topreg[:, 0], topreg[:, 1], color="r")
plt.show()
