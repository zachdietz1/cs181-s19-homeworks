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
        self.mus  = [None for i in range(3)]
        self.sigmas  = [None for i in range(3)]
        self.pis = [None for i in range(3)]
    
    # Just to show how to make 'private' methods
    def __dummy_private_method(self, input):
        return None

    def __calculate_mu(self, X):
        n, mu = X.shape[0], [0,0]
        for x in X:
            mu += x / n
        return mu

    def __calculate_covariance(self, X, separate = True, y = None):
        covariance, n = np.zeros(np.dot(X[0],X[0].T).shape), X.shape[0]
        mu = self.__calculate_mu(X)
        
        # separate covariances
        if separate:
            for i, x in enumerate(X):
                difference = (x - mu).reshape(2,1) 
                val = np.matmul(difference, difference.T) / n
                covariance = np.add(covariance, val)

        # shared covariances
        else:
            for i, x in enumerate(X):
                difference = (x - self.mus[y[i]]).reshape(2,1)
                val = np.matmul(difference, difference.T) / n
                covariance = np.add(covariance, val)
        
        return covariance

    def fit(self, X, y):
        data_by_class = [[],[],[]]
        
        # separate the data by the classes 
        for i, x in enumerate(X):
            data_by_class[y[i]].append(x)

        # now turn them into np.arrays
        for i, _class in enumerate(data_by_class):
            data_by_class[i] = np.array(data_by_class[i])

        # calculate the mus and priors for each class
        n = X.shape[0]
        for i, class_ in enumerate(data_by_class):
            self.mus[i] = self.__calculate_mu(class_)
            self.pis[i] = class_.shape[0] / n 

        if not(self.is_shared_covariance):
            # covariance should be the same for each class
            for i, class_ in enumerate(data_by_class):
                self.sigmas[i] = self.__calculate_covariance(class_)
          
        else:
            # covariance should be different
            cov = self.__calculate_covariance(X, False, y)
            for i in range(len(self.sigmas)):
                self.sigmas[i] = cov

        return

    def predict(self, X_pred):
        preds = []
        for x in X_pred:
            guesses = []
            for i, mu in enumerate(self.mus):
                guesses.append(mvn.pdf(x, mu, self.sigmas[i]) * self.pis[i])
            preds.append(np.argmax(guesses))
        return np.array(preds)

    def negative_log_likelihood(self, X, y):
        loss = 0
        for i, x in enumerate(X):
            loss += np.log(mvn.pdf(x,self.mus[y[i]],self.sigmas[y[i]]) * self.pis[y[i]])
        return (- loss)

'''
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance
        self.means, self.covs, self.pis = [None, None, None], [None, None, None], [None, None, None]
    # Just to show how to make 'private' methods
    def __dummy_private_method(self, input):
        return None

    def __get_mean(self, X):
        # X should come in as a n x 2 vector
        n = X.shape[0]
        mean = [0,0]
        for x_i in X:
            mean += x_i / n
        return mean

    def __get_cov(self, X, share = False, y = None):
        # X should come in as a n x 2 vector
        cov = np.zeros(np.dot(X[0],X[0].T).shape)
        n = X.shape[0]
        mu = self.__get_mean(X)
        for i, x_i in enumerate(X):
            if share:
                dif = (x_i - self.means[y[i]]).reshape(2,1)
            else:
                dif = (x_i - mu).reshape(2,1)
            sum_term = np.matmul(dif, dif.T) / n
            cov = np.add(cov, sum_term)
        return cov

    # TODO: Implement this method!
    def fit(self, X, y):
        # separate out the classes
        separated_by_class = [[],[],[]]
        for i, x_i in enumerate(X):
            separated_by_class[y[i]].append(x_i)

        # turn the separated classes into numpy objects
        for i, _class in enumerate(separated_by_class):
            separated_by_class[i] = np.array(separated_by_class[i])

        # get mean for each class
        for i, class_ in enumerate(separated_by_class):
            self.means[i] = self.__get_mean(class_)

        # get pis
        n = X.shape[0]
        for i, class_ in enumerate(separated_by_class):
            self.pis[i] = class_.shape[0] / n 

        if self.is_shared_covariance:
            cov = self.__get_cov(X, True, y)
            for i in range(3):
                self.covs[i] = cov
        else:
            for i, class_ in enumerate(separated_by_class):
                self.covs[i] = self.__get_cov(class_)

        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        preds = []
        for x in X_pred:
            max_ = -1
            index = -999
            for i, mu in enumerate(self.means):
                prob = mvn.pdf(x, mu, self.covs[i]) * self.pis[i]
                if prob > max_:
                    max_ = prob
                    index = i
            preds.append(index)
        return np.array(preds)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        nll = 0
        for i, x_i in enumerate(X):
            nll = nll - np.log(mvn.pdf(x_i,self.means[y[i]],self.covs[y[i]]) * self.pis[y[i]])
        return nll
    '''