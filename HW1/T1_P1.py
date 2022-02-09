#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import numpy as np

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]

# separate data into list of inputs/outputs
x = []
y = []
for a,b in data:
    x.append(a)
    y.append(b)

def compute_loss(tau):
    loss = 0
    for n in range(0, len(data)):
        y_est = 0
        for m in range(0, len(data)):
            if (m != n):
                y_est += np.exp(-(x[m] - x[n])**2/tau) * y[m]
        loss += (y[n] - y_est)**2
    return loss

for tau in (0.01, 2, 100, ):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))