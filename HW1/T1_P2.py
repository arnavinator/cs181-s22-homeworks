#####################
# CS 181, Spring 2022
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# set up data
data = [(0., 0.),
        (1., 0.5),
        (2., 1),
        (3., 2),
        (4., 1),
        (6., 1.5),
        (8., 0.5)]

x_train = np.array([d[0] for d in data])
y_train = np.array([d[1] for d in data])

x_test = np.arange(0, 12, .1)

print("y is:")
print(y_train)

# def predict_knn(k=1, tau=1):
#     """Returns predictions for the values in x_test, using KNN predictor with the specified k."""
#     y_test = [] # list to return
#     for x in x_test:
#         # find k nearest inputs to x
#         # first, calculate dist between x and all data
#         dist = {}  # dictionary to store (distance : ydata pairs)
#         for xdata, ydata in data:
#             distance = np.exp(-(xdata-x)**2/tau)
#             dist[distance] = ydata
#
#         print(len(dist))
#         # sort by distance, include top k, and predict
#         sorted_keys = sorted(dist)
#         predict = 0
#         for i in range(0, k):
#             predict += dist[sorted_keys[i]]
#             # print(i)
#             # print(x)
#             # print(sorted_keys)
#             # print(dist)
#
#         # append the prediction to y_test
#         y_test.append(predict/k)
#
#     return y_test

# used for sorting the list later on
def sortFirst(e):
    return e[0]

def predict_knn(k=1, tau=1):
    """Returns predictions for the values in x_test, using KNN predictor with the specified k."""
    y_test = []  # list to return
    for x in x_test:
        # find k nearest inputs to x
        # first, calculate dist between x and all data
        dist = [] # list to store (distance, ydata pairs)
        for xdata, ydata in data:
            distance = np.exp(-(xdata - x) ** 2 / tau)
            dist.append([distance, ydata])

        # sort by distance, include top k, and predict
        dist.sort(key=sortFirst)
        predict = 0
        for i in range(0, k):
            predict += dist[i][1]

        # append the prediction to y_test
        y_test.append(predict / k)

    return y_test

def plot_knn_preds(k):
    plt.xlim([0, 12])
    plt.ylim([0,3])
    
    y_test = predict_knn(k=k)
    
    plt.scatter(x_train, y_train, label = "training data", color = 'black')
    plt.plot(x_test, y_test, label = "predictions using k = " + str(k))

    plt.legend()
    plt.title("KNN Predictions with k = " + str(k))
    plt.savefig('k' + str(k) + '.png')
    plt.show()

for k in (1, 3, len(x_train)-1):
    plot_knn_preds(k)
