import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def fit(self, x, y):
        # FITTING BASED ON MY PSET 2 ANSWERS
        n = len(y)
        # calculate z_k (sum of total data points which correspond to class k)
        # and calculate sum of x in each class (for Gaussian mean calculation)
        z_k = np.array([0., 0., 0.])
        self.u_k = np.array([[0.,0.], [0.,0.], [0.,0.]])
        for i in range(0, n):
            if y[i] == 0:
                z_k[0] += 1
                self.u_k[0] += x[i]
            elif y[i] == 1:
                z_k[1] += 1
                self.u_k[1] += x[i]
            else:
                z_k[2] += 1
                self.u_k[2] += x[i]

        # class prior pi_k
        self.pi_k = z_k / n

        # if input x is 27x2, each data point is R^2, then u_k is R^2 as well
        # u_k Gaussian mean for each class... 3x2 matrix, row 1 is u_1
        for i in range(0, 3):
            self.u_k[i] = self.u_k[i] / z_k[i]


        if self.is_shared_covariance:
            # shared covariance is the formula I derived in pset 2, q2
            # x-u_k is 1x2, so (x-uk).T * (x-uk) is a 2x2 matrix
            self.cov = np.array([[0.,0.],[0.,0.]])
            for i in range(0, n):
                if y[i] == 0:
                    vec = x[i] - self.u_k[0]
                elif y[i] == 1:
                    vec = x[i] - self.u_k[1]
                else:
                    vec = x[i] - self.u_k[2]
                # add to covariance term
                vec = np.reshape(vec, (vec.shape[0], 1))
                self.cov += np.dot(vec, vec.T)
            self.cov = self.cov/n
        else:
            # same as shared covariance, but don't add for each class
            # and divide covk by z_k (# of points in class k) instead of dividing by n (total points)
            self.cov0 = np.array([[0., 0.], [0., 0.]])
            self.cov1 = np.array([[0., 0.], [0., 0.]])
            self.cov2 = np.array([[0., 0.], [0., 0.]])
            for i in range(0, n):
                if y[i] == 0:
                    vec = x[i] - self.u_k[0]
                    vec = np.reshape(vec, (vec.shape[0], 1))
                    self.cov0 += np.dot(vec, vec.T)
                elif y[i] == 1:
                    vec = x[i] - self.u_k[1]
                    vec = np.reshape(vec, (vec.shape[0], 1))
                    self.cov1 += np.dot(vec, vec.T)
                else:
                    vec = x[i] - self.u_k[2]
                    vec = np.reshape(vec, (vec.shape[0], 1))
                    self.cov2 += np.dot(vec, vec.T)
            self.cov0 = self.cov0 / z_k[0]
            self.cov1 = self.cov1 / z_k[1]
            self.cov2 = self.cov2 / z_k[2]



    def predict(self, x):
        if self.is_shared_covariance:
            y_out = []
            for elem in x:
                p1 = self.pi_k[0] / ((2 * np.pi) * np.linalg.norm(self.cov)**0.5) * np.exp(
                    -0.5 * (elem - self.u_k[0]) @ np.linalg.inv(self.cov) @ (elem - self.u_k[0]).T)
                p2 = self.pi_k[1] / ((2 * np.pi) * np.linalg.norm(self.cov)**0.5) * np.exp(
                    -0.5 * (elem - self.u_k[1]) @ np.linalg.inv(self.cov) @ (elem - self.u_k[1]).T)
                p3 = self.pi_k[2] / ((2 * np.pi) * np.linalg.norm(self.cov)**0.5) * np.exp(
                    -0.5 * (elem - self.u_k[2]) @ np.linalg.inv(self.cov) @ (elem - self.u_k[2]).T)

                if p1 >= p2 and p1 >= p3:
                    y_out.append(0)
                elif p2 >= p3 and p2 >= p1:
                    y_out.append(1)
                else:
                    y_out.append(2)
            return np.array(y_out)
        else:
            y_out = []
            for elem in x:
                p1 = self.pi_k[0] / ((2 * np.pi) * np.linalg.norm(self.cov0)**0.5) * np.exp(
                    -0.5 * (elem - self.u_k[0]) @ np.linalg.inv(self.cov0) @ (elem - self.u_k[0]).T)
                p2 = self.pi_k[1] / ((2 * np.pi) * np.linalg.norm(self.cov1)**0.5) * np.exp(
                    -0.5 * (elem - self.u_k[1]) @ np.linalg.inv(self.cov1) @ (elem - self.u_k[1]).T)
                p3 = self.pi_k[2] / ((2 * np.pi) * np.linalg.norm(self.cov2)**0.5) * np.exp(
                    -0.5 * (elem - self.u_k[2]) @ np.linalg.inv(self.cov2) @ (elem - self.u_k[2]).T)

                if p1 >= p2 and p1 >= p3:
                    y_out.append(0)
                elif p2 >= p3 and p2 >= p1:
                    y_out.append(1)
                else:
                    y_out.append(2)
            return np.array(y_out)


    def negative_log_likelihood(self, x, y):
        loss = 0
        if self.is_shared_covariance:
            for i in range(0, len(y)):
                loss += -np.log(self.pi_k[y[i]]) - np.log(1 / (2 * np.pi * np.linalg.norm(self.cov) ** 0.5)) + 0.5 * (
                                x[i] - self.u_k[y[i]]) @ np.linalg.inv(self.cov) @ (x[i] - self.u_k[y[i]]).T
            return loss
        else:
            for i in range(0, len(y)):
                if y[i] == 0:
                    loss += -np.log(self.pi_k[y[i]]) - np.log(
                        1 / (2 * np.pi * np.linalg.norm(self.cov0) ** 0.5)) + 0.5 * (
                                    x[i] - self.u_k[y[i]]) @ np.linalg.inv(self.cov0) @ (x[i] - self.u_k[y[i]]).T
                elif y[i] == 1:
                    loss += -np.log(self.pi_k[y[i]]) - np.log(
                        1 / (2 * np.pi * np.linalg.norm(self.cov1) ** 0.5)) + 0.5 * (
                                    x[i] - self.u_k[y[i]]) @ np.linalg.inv(self.cov1) @ (x[i] - self.u_k[y[i]]).T
                else:
                    loss += -np.log(self.pi_k[y[i]]) - np.log(
                        1 / (2 * np.pi * np.linalg.norm(self.cov2) ** 0.5)) + 0.5 * (
                                    x[i] - self.u_k[y[i]]) @ np.linalg.inv(self.cov2) @ (x[i] - self.u_k[y[i]]).T
            return loss


