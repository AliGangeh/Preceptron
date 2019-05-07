#imports libraries
import numpy as np
import matplotlib.pyplot as plt

#plots lines wait for fraction of second then removes it
def draw(x1,x2):
    ln=plt.plot(x1,x2)
    plt.pause(0.0001)
    ln[0].remove()

#determines chance of diabetes plugging in score in the sigmoid function, this turns
#the score into a decimal between 0 and 1. The higher it is the healthier they are.
def sigmoid(score):
    return 1/(1+np.exp(-score)) #sigmoid function

#calculates how much error there is with the current line.
#it does this through cross entropy here is a good source on how it works
# https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
#cross entropy gives an value between 1 and 0, the higher the number the higher the error
def calculate_error(line_parameters, points , y):
    #points is an array with all the points and their coordinates. along with the bias multiplier
    #which is 1, this creates the shape (2*n_pts, 3) n accesses the item at index 0, 2*n_pts
    n=points.shape[0]
    #runs sigmoid function whit the points (2*n_pts, 3) multiplied by the line (3,1) this spits out
    # (2_pts, 1) matrix with the error for each point
    p= sigmoid(points*line_parameters)
    #look at general description of function
    cross_entropy=-(1/n)*(np.log(p).T * y + np.log(1 - p).T * (1 - y))
    return cross_entropy

def grad_descent(line_parameters, points, y, alpha):
    n = points.shape[0]
    #determines number of iterations, the higher the iterations the more accurate it is.
    for i in range(2000):
        #runs sigmoid function
        p = sigmoid(points*line_parameters)
        #finds the gradient (2*n_pts, 3) transposed and multiplied by prediction subtracted by true
        #value. multiplied by the learning rate over 200
        gradient = (points.T * (p - y)) * (alpha / n)
        #the line is changed by subtracting the gradient from it. getting it closer to the best value
        line_parameters= line_parameters - gradient
        #defines and draws lines
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        x1 = np.array([points[:,0].min(), points[:,0].max()])
        x2 = -b/w2 + (x1*(-w1/w2))
        draw(x1, x2)
        #prints error
        print(calculate_error(line_parameters, points, y))

#creates 100pts
n_pts=100
#keeps same randomized set same every time
np.random.seed(0)
#creates a 1d array (100)
bias = np.ones(n_pts)
#creates the sick patient, example dataset
#generates 100 random points averaging around point of (12,10). This actually
#creates 2d array with the shape (npts,3) for the 3 values in the width there is x,y
#and the bias... currently set as one by the bias variable this is the b in y=mx+b
#this is all then transposed so that its not (3,100) but (100,3) this is important
#for matrix multiplication
top_region = np.array([np.random.normal(10,2,n_pts), np.random.normal(12,2,n_pts), bias]).T
#does the same thing as above except for the second group of healthy people
bottom_region = np.array([np.random.normal(5,2, n_pts), np.random.normal(6,2, n_pts), bias]).T
#combines the points into one array with a shape of (2*npts, 3)
all_points = np.vstack((top_region, bottom_region))
#creates the starting point for the line it is a matrix of 0s and transposes it so it is (3,1)
#in shape
line_parameters = np.matrix([np.zeros(3)]).T
#multiplies weights by the parameters, matrix multiplication this spits out a value
#we will call score, when score is negative that means the algorithem thinks their sick
#if their positive then they are healthy. this is plugged into the sigmoid funcition
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)

#determines size of graph
_, ax= plt.subplots(figsize=(8, 8))
#displays sick ppl in red and healthy in blue
ax.scatter(top_region[:,0], top_region[:,1], color='r')
ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')
# runs gradient descent with learning rate of 0.06.
grad_descent(line_parameters, all_points, y, 0.06)
#shows final graph with divisor
plt.show()
