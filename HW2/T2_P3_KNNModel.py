import numpy as np

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.x = None
        self.y = None
        self.K = k

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def predict(self, x):
        y_out = []
        for star in x:
            # find k nearest inputs to x
            # first, calculate dist between x and all data
            dist = []
            for i in range(0, len(self.y)):
                # store dist between star and training data, and output label
                dist.append([self.dist(star, self.x[i]), self.y[i]])

            # sort from smallest to largest dist, include top k, and predict
            nearest = 0
            y_votes = []
            while (nearest < self.K):
                min = dist[0]
                # find smallest distance in list
                for elem in dist:
                    if min[0] >= elem[0]:
                        min = elem
                # append the label of this mimimum, and remove from dist
                y_votes.append(min[1])
                dist.remove(min)
                nearest += 1

            # now take majority vote to decide label
            if y_votes.count(0) > y_votes.count(1) and y_votes.count(0) > y_votes.count(2):
                y_out.append(0)
            elif y_votes.count(1) > y_votes.count(0) and y_votes.count(1) > y_votes.count(2):
                y_out.append(1)
            else:
                y_out.append(2)

        return np.array(y_out)

    # distance between two stars, where a,b are (mag,temp) pairs
    def dist(self, a, b):
        return (((a[0] - b[0])/3)**2 + (a[1] - b[1])**2)

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, x, y):
        self.x = x
        self.y = y