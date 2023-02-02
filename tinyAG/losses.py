import numpy as np
import math

class losses:

    def __init__(self, func, name=None):
        self.name = name
        self.func = func
    
    def __call__(self, ytrue, ypred): 
        return self.func(ytrue, ypred) 

class MSE(losses):

    def __init__(self, name = 'mean_squared_error'):
        super().__init__(mean_squared_error, name = name)

class MAE(losses):

    def __init__(self, name = 'mean_absolute_error'):
        super().__init__(mean_absolute_error, name = name)

def mean_squared_error(ytrue, ypred):
    return sum((yp - yt)**2 for yt, yp in zip(ytrue, ypred))
    
def mean_absolute_error(ytrue, ypred):
    return sum(abs(yp - yt) for yt, yp in zip(ytrue, ypred))