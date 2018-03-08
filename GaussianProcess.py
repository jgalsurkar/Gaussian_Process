import numpy as np
import pandas as pd
import itertools as it
import scipy
import scipy.spatial

class rbf(object):
    """Radial Basis Function (RBF/Gaussian Kernel)
    
    Parameters
    ----------        
    param_b : {Integer, Float}
        rbf parameter, controls kernel width
    
    param_a : {Integer, Float}
        rbf alpha parameter
    """
    def __init__(self, param_b, param_a = 1):
        self.param_b = param_b
        self.param_a = param_a

    def compute(self, pointa, pointb):
        norm = scipy.spatial.distance.cdist(pointa, pointb, 'euclidean')
        return self.param_a * np.exp((-1/self.param_b)*np.square(norm))

class GaussianProcess(object):
    """Gaussian Process Model for Regression
    
    Parameters
    ----------        
    x_train : {array-like, sparse matrix},
    shape = [n_samples, n_features]
        Training data
    
    y_train : array-like, shape = [n_samples]
        Training targets
    
    kernel : callable
        kernel to be used
        
    sigma : float
        Amplitude to control scaling
    """

    def __init__(self, x_train, y_train, kernel, sigma = .1):
        self.X = x_train
        self.y = y_train
        self.sigma = sigma
        self.kernel = kernel
        self.kernal_covariance = self.kernel.compute(self.X, self.X)
        self.results = {}

    def updateSigma(self, sigma):
        """Update sigma value.
        We do this to re train the model without having to keep re-compute the covariance matrix

        Parameters
        ----------      
        sigma : float
            Amplitude to control scaling
        """
        self.sigma = sigma

    def predict_single(self, new_point):
        """Compute the predicted mean and variance to be used to learn the overall distribution

        Parameters
        ----------      
        new_point : array-like, shape = [1, n_features]
            Single data point
            
        Returns
        ----------      
        Tuple : [Float, Float]
            Mean and Variance
        """
        new_point = np.asmatrix(new_point)
        new_point_kernel = np.array(self.kernel.compute(new_point, self.X).T)
        noise_kernel = self.sigma*np.eye(self.kernal_covariance.shape[0]) + self.kernal_covariance
        kernel_mult = np.dot(np.linalg.inv(noise_kernel), new_point_kernel)
        
        mean = np.ravel(np.dot(kernel_mult.T, self.y))
        variance = np.ravel(self.sigma + self.kernel.compute(new_point, new_point) - np.dot(kernel_mult.T, new_point_kernel))
        
        return [mean, variance]
    
    def trackResults(self, predictions):
        """Track the mean and variance for each sigma

        Parameters
        ----------      
        predictions : List[(Mean, Variance)]
            List of mean and variance for ever 
            
        Returns
        ----------      
        Tuple : [Float, Float]
            Mean and Variance
        """
        means = np.array([x[0] for x in predictions])
        variances = np.array([x[1] for x in predictions])
        self.results[self.sigma] = [means, variances]
    
    def predict(self, x_test):
        """Learn the gaussian approximation of the test set

        Parameters
        ----------      
        test_set : {array-like, sparse matrix},
        shape = [n_samples, n_features]
            Test set
        """
        predictions = [self.predict_single(x_test.loc[i]) for i in range(len(x_test))] 
        self.trackResults(predictions)