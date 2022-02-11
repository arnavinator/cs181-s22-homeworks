#####################
# CS 181, Spring 2022
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
# plt.figure(1)
# plt.plot(years, republican_counts, 'o')
# plt.xlabel("Year")
# plt.ylabel("Number of Republicans in Congress")
# plt.figure(2)
# plt.plot(years, sunspot_counts, 'o')
# plt.xlabel("Year")
# plt.ylabel("Number of Sunspots")
# plt.figure(3)
# plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
# plt.xlabel("Number of Sunspots")
# plt.ylabel("Number of Republicans in Congress")
# plt.show()


# Create the simplest basis, with just the time and an offset
# Matrix with 2 columns, the first which is all 1s
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
    res = []    # matrix to return
    xx = np.array(xx)

    if part == 'a' and is_years:  # takes X array, and subtract 1960 from all entry, then div by 40
        xx = (xx - np.array([1960] * len(xx))) / 40

    if part == "a" and not is_years:  # takes X array, and divide
        xx = xx/20

    if part == 'a':
        # x^1 ... x^j
        res.append(xx)
        res.append(xx * xx)
        res.append(xx * xx * xx)
        res.append(xx * xx * xx * xx)
        res.append(xx * xx * xx * xx * xx)

    if part == 'b':
        for uj in range(1960, 2011, 5):
            res.append(np.exp(-(xx - uj)**2 / 25))

    if part == 'c':
        for j in range(1, 6):
            res.append(np.cos(xx/j))

    if part == 'd':
        for j in range(1, 26):
            res.append(np.cos(xx/j))
        
    return np.array(res)

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

def sse(years, grid_years, grid_Yhat, train_y):
    sum = 0
    # for every elem in years
    for i in range(0, len(train_y)):
        # find the closest match to year in grid_years, and get its index
        matches = np.where((grid_years >= years[i]-0.5) & (grid_years <= years[i]+0.5))
        if len(matches[0]) != 0:
            index = matches[0][0] + int(len(matches)/2)  # this is closest index for grid_year to year
            # compute sse between elem and grid_Yhat[index]
            sum += (train_y[i] - grid_Yhat[index])**2
    return sum




# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
# default line of best fit with no basis regression
grid_years = np.linspace(1960, 2005, 200)  # X-TEST... all x years we want to predict
grid_X = np.vstack((np.ones(grid_years.shape), grid_years))   # X-TEST, but matrix... 2 rows (1st all 1s)
grid_Yhat  = np.dot(grid_X.T, find_weights(X, Y))   # THIS IS Y-TEST for no basis regression


### REPUBLICANS  VS   YEARS  ###
# plot of basis expansion PART A
X_a= np.vstack((np.ones(years.shape), make_basis(years, part='a', is_years=True))).T
grid_X_a = np.vstack((np.ones(grid_years.shape), make_basis(grid_years, part='a', is_years=True)))
grid_Yhat_a = np.dot(grid_X_a.T, find_weights(X_a, Y))

# plot of basis expansion PART B
X_b= np.vstack((np.ones(years.shape), make_basis(years, part='b', is_years=True))).T
grid_X_b = np.vstack((np.ones(grid_years.shape), make_basis(grid_years, part='b', is_years=True)))
grid_Yhat_b = np.dot(grid_X_b.T, find_weights(X_b, Y))

# plot of basis expansion PART C
X_c= np.vstack((np.ones(years.shape), make_basis(years, part='c', is_years=True))).T
grid_X_c = np.vstack((np.ones(grid_years.shape), make_basis(grid_years, part='c', is_years=True)))
grid_Yhat_c = np.dot(grid_X_c.T, find_weights(X_c, Y))

# plot of basis expansion PART D
X_d= np.vstack((np.ones(years.shape), make_basis(years, part='d', is_years=True))).T
grid_X_d = np.vstack((np.ones(grid_years.shape), make_basis(grid_years, part='d', is_years=True)))
grid_Yhat_d = np.dot(grid_X_d.T, find_weights(X_d, Y))

# SSE for each part, REPUBLICANS
print("REPUBLICANS VS YEARS SSE")
print("SSE for regular basis: ", sse(years, grid_years, grid_Yhat, republican_counts))
print("SSE for basis (a): ", sse(years, grid_years, grid_Yhat_a, republican_counts))
print("SSE for basis (b): ", sse(years, grid_years, grid_Yhat_b, republican_counts))
print("SSE for basis (c): ", sse(years, grid_years, grid_Yhat_c, republican_counts))
print("SSE for basis (d): ", sse(years, grid_years, grid_Yhat_d, republican_counts))

#
# # Plot the data and the regression line for REPUBLICANS.
# plt.figure(1)
# plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat_a, '-')  # PLOT TRAIN AND TEST
# plt.xlabel("Year")
# plt.ylabel("Number of Republicans in Congress")
# plt.title("Basis A")
# plt.figure(2)
# plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat_b, '-')  # PLOT TRAIN AND TEST
# plt.xlabel("Year")
# plt.ylabel("Number of Republicans in Congress")
# plt.title("Basis B")
# plt.figure(3)
# plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat_c, '-')  # PLOT TRAIN AND TEST
# plt.xlabel("Year")
# plt.ylabel("Number of Republicans in Congress")
# plt.title("Basis C")
# plt.figure(4)
# plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat_d, '-')  # PLOT TRAIN AND TEST
# plt.xlabel("Year")
# plt.ylabel("Number of Republicans in Congress")
# plt.title("Basis D")
# plt.show()

#

### REPUBLICANS VS SUNSPOTS ###
Y = republican_counts[years<last_year]

grid_spots = np.linspace(0, 160, 200)  # X-TEST
# grid_X = np.vstack((np.ones(grid_spots.shape), grid_spots))   # X-TEST, but matrix


X_a= np.vstack((np.ones(sunspot_counts[years<last_year].shape), make_basis(sunspot_counts[years<last_year], part='a', is_years=False))).T
grid_X_a = np.vstack((np.ones(grid_spots.shape), make_basis(grid_spots, part='a', is_years=False)))
grid_Yhat_a = np.dot(grid_X_a.T, find_weights(X_a, Y))

# plot of basis expansion PART C
X_c= np.vstack((np.ones(sunspot_counts[years<last_year].shape), make_basis(sunspot_counts[years<last_year], part='c', is_years=False))).T
grid_X_c = np.vstack((np.ones(grid_spots.shape), make_basis(grid_spots, part='c', is_years=False)))
grid_Yhat_c = np.dot(grid_X_c.T, find_weights(X_c, Y))

# plot of basis expansion PART D
X_d= np.vstack((np.ones(sunspot_counts[years<last_year].shape), make_basis(sunspot_counts[years<last_year], part='d', is_years=False))).T
grid_X_d = np.vstack((np.ones(grid_spots.shape), make_basis(grid_spots, part='d', is_years=False)))
grid_Yhat_d = np.dot(grid_X_d.T, find_weights(X_d, Y))


# SSE for each part, REPUBLICANS
print("REPUBLICANS VS SUNSPOTS SSE")
print("SSE for basis (a): ", sse(sunspot_counts[years<last_year], grid_spots, grid_Yhat_a, republican_counts[years<last_year]))
print("SSE for basis (c): ", sse(sunspot_counts[years<last_year], grid_spots, grid_Yhat_c, republican_counts[years<last_year]))
print("SSE for basis (d): ", sse(sunspot_counts[years<last_year], grid_spots, grid_Yhat_d, republican_counts[years<last_year]))


# Plot the data and the regression line for REPUBLICANS.
plt.figure(1)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o', grid_spots, grid_Yhat_a, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.title("Basis A")
plt.figure(2)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o', grid_spots, grid_Yhat_c, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.title("Basis C")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o', grid_spots, grid_Yhat_d, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.title("Basis D")
plt.show()