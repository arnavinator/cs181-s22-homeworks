import numpy as np
import matplotlib.pyplot as plt



# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.runs = 200000
        self.loss_index = []
        self.loss = []

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # we need weights for each class (3 sets of weights)
    def fit(self, x, y, w_init=None):
        # add a bias of 1s to x
        x = np.concatenate((np.ones(x.shape[0]).reshape(x.shape[0], 1), x), axis=1)
        # given x is Nx2 mat, then we want wT to be 2x1 mat
        # then, wT * x can be expressed as np.dot(x, w) to get result of Nx1 matrix
        # which represents the logistic probability for each input elem of x
        # randomize a set of arrays of the right size
        self.W1 = np.random.rand(x.shape[1], 1)
        self.W2 = np.random.rand(x.shape[1], 1)
        self.W3 = np.random.rand(x.shape[1], 1)
        self.gradientDescent(x, y)

    # perform gradient descent
    # given some weights, predict output
    # 'Dwarf': 0,       # also corresponds to 'red' in the graphs
    # 'Giant': 1,       # also corresponds to 'blue' in the graphs
    # 'Supergiant': 2   # also corresponds to 'green' in the graphs
    def gradientDescent(self, x, y):
        # modify y for each class gradient descent loss
        y1 = []   # 1 if dwarf, 0 else
        y2 = []   # 1 if giant, 0 else
        y3 = []   # 1 if supergiant, 0 else
        for elem in y:
            if elem == 0:
                y1.append([1])
                y2.append([0])
                y3.append([0])
            elif elem == 1:
                y1.append([0])
                y2.append([1])
                y3.append([0])
            elif elem == 2:
                y1.append([0])
                y2.append([0])
                y3.append([1])
        y1 = np.array(y1)
        y2 = np.array(y2)
        y3 = np.array(y3)

        for i in range(0, self.runs):
            y1_pred = self.descent_predict(x,clasif=0)
            y2_pred = self.descent_predict(x,clasif=1)
            y3_pred = self.descent_predict(x,clasif=2)
            self.W1 = self.W1 - self.eta*(np.dot(x.T, (y1_pred- y1)) + self.lam*self.W1)
            self.W2 = self.W2 - self.eta*(np.dot(x.T, (y2_pred- y2)) + self.lam*self.W2)
            self.W3 = self.W3 - self.eta*(np.dot(x.T, (y3_pred- y3)) + self.lam*self.W3)

            self.update_loss(i, y, y1_pred, y2_pred, y3_pred)


    def descent_predict(self, x, clasif):
        res1 = np.exp(np.dot(x, self.W1))  # this is a Nx1 matrix
        res2 = np.exp(np.dot(x, self.W2))
        res3 = np.exp(np.dot(x, self.W3))
        sum = res1 + res2 + res3
        if clasif == 0:
            return res1/sum
        elif clasif == 1:
            return res2/sum
        else:
            return res3/sum

    def update_loss(self, i, y, y1_pred, y2_pred, y3_pred):
        self.loss_index.append(i)
        loss = 0
        # calculate negative log likelihood (aka cross-entropy error)
        for i in range(0, len(y)):
            if y[i] == 0:
                loss += -1*np.log(y1_pred[i])
            elif y[i] == 1:
                loss += -1*np.log(y2_pred[i])
            else:
                loss += -1*np.log(y3_pred[i])
        self.loss.append(loss)

    # given some weights, predict output
    # 'Dwarf': 0,       # also corresponds to 'red' in the graphs
    # 'Giant': 1,       # also corresponds to 'blue' in the graphs
    # 'Supergiant': 2   # also corresponds to 'green' in the graphs
    def predict(self, x):
        # add bias term once again
        x = np.concatenate((np.ones(x.shape[0]).reshape(x.shape[0],1), x), axis=1)
        res1 = np.exp(np.dot(x, self.W1))  # this is a Nx1 matrix
        res2 = np.exp(np.dot(x, self.W2))
        res3 = np.exp(np.dot(x, self.W3))

        output = []
        # use probabilities to make predictions
        for i in range(0, x.shape[0]):
            if res1[i] >= res2[i] and res1[i] >= res3[i]:
                output.append(0)
            elif res2[i] >= res1[i] and res2[i] >= res3[i]:
                output.append(1)
            else:
                output.append(2)
        return np.array(output)


    def visualize_loss(self, output_file, show_charts=False):
        # for each of the 200000 iterations, calculate the loss
        # output_file is just the title we want
        # we need to actually do plt in this function
        # make some tracking mechanism as the algo progresses
        plt.figure()
        plt.scatter(self.loss_index, self.loss, alpha=0.5)
        plt.title(output_file)
        plt.xlabel('Number of Iterations')
        plt.ylabel('NegativeLog-Likelihood  Loss')
        if show_charts:
            plt.show()
