from typing import Tuple
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, alpha=0.0001, iteration=10000, error = 0.0001):
            self.alpha = alpha
            self.iteration = iteration
            self.error = error
            
        # the arguments are ignored anyway, so we make them optional
    def fit(self, X, y):
        beta=np.random.randn(X.shape[1],1)
        intecept=0
        cost = 999999
        iters = 1
        
        while iters < self.iteration and cost > self.error:
            
            gradients = np.dot(X.T, (np.dot(X,beta)-y) )
#             print(X.shape,' X', gradients.shape,'G', y.shape)
            beta -= self.alpha * gradients
            
#             intercept -= self.alpha* gradients
            cost = 1/2 * np.sum((np.dot(X, beta)-y) ** 2)
        
        self.beta = beta
        print(self.beta)
#         self.intercept = intercept
       
    def transform(self, X):
        return np.dot(X, self.beta)