import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
        self.W = np.random.rand(3,3)
        self.epochs = 1000
        self.errors = [None for i in range(self.epochs)]

    # Just to show how to make 'private' methods
    def __dummy_private_method(self, input):
        return None


    def __log_soft_max(self, X):
        return np.subtract(X, misc.logsumexp(X))

    def __soft_max(self, X):
        return np.exp(self.__log_soft_max(X))


    def __error(self, X, y):
        error = 0
        for i, x_i in enumerate(X):
            y_ij_hat = self.__soft_max(np.dot(self.W,x_i.T))[y[i]]
            error = error - np.log(y_ij_hat)
        return error

    def fit_once_biased(self, X, y):
        # Perform gradient descent
        for k in range(self.epochs):
            W_dot_X = np.dot(self.W,X.T).T # 3 x 27
            # apply soft max
            y_hatss = np.array([self.__soft_max(y) for y in W_dot_X]).T # 3 x 27 : classes x data length
            # loop through all three classes
            for i, y_hats in enumerate(y_hatss):
                # initialize the gradient to lambda*W_i
                grad = self.lambda_parameter * self.W[i]

                # loop through each of the 27 predictions within each class
                for j, y_hat in enumerate(y_hats):
                    # calculate gradient
                    grad += (y_hat - (y[j]==i)) * X[j]
                self.W[i] = self.W[i] - self.eta * grad
            self.errors[k] = self.__error(X, y)
            #print(self.errors)
        return

    def fit(self, X, y):
        ones_added = np.array([np.append(column, [1]) for column in X])
        self.fit_once_biased(ones_added, y)
        print(self.W)

    # TODO: Implement this method!
    def predict(self, X_pred):
        preds = []
        for x in X_pred:
            one_added = np.append(x,[1])
            y_hat = self.__soft_max(np.dot(self.W,one_added.T))
            
            max_, index = -1, -999
            for i, num in enumerate(y_hat):
                if num > max_:
                    max_ = num
                    index = i 
            preds.append(index)
        return np.array(preds)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        plt.figure()
        plt.plot(np.arange(self.epochs), self.errors)
        plt.suptitle("Negative Log Likelihood Loss with Eta={} and Lambda={}".format(self.eta,self.lambda_parameter))
        plt.xlabel("Epochs")
        plt.ylabel("Neg. Log Likelihood Loss")
        plt.show()
        #print(self.errors[-1])
