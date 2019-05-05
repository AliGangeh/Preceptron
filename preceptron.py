import numpy as np
import matplotlib.pyplot as plt

def draw(x1, x2):
    ln=plt.plot(x1,x2)

def sigmoid(score):
    return 1/(1+np.exp(-score))

#data points to be classified
npts=100
np.random.seed(0)
bias=np.ones(npts)
topreg=np.array([np.random.normal(10, 2, npts), np.random.normal(12,2,npts), bias]).T
botreg=np.array([np.random.normal(5,2, npts), np.random.normal(6,2,npts), bias]).T
allpts=np.vstack((topreg, botreg))

w1= -0.2
w2= -0.35
b=3.5
linepara=np.matrix([w1,w2,b]).T
x1 = np.array([botreg[:, 0].min(), topreg[:, 1].max()])
x2 = -b / w2 + x1 * (-w1 / w2)
linearcombo=allpts*linepara
prob=sigmoid(linearcombo)
print(prob)

_, ax = plt.subplots(figsize=(4,4))
ax.scatter(botreg[:, 0], botreg[:, 1], color="b")
ax.scatter(topreg[:, 0], topreg[:, 1], color="r")
draw(x1,x2)
plt.show()
