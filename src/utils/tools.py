import numpy as np
import torch

class MinMaxScaler:
    # data: np.2darray
    def __init__(self, lower=-1, upper=1):
        self.lower = lower
        self.upper = upper

    def fit(self, data):
        self.max = data.max(0)
        self.min = data.min(0)

    def transform(self, data):
        return (self.max - data) / (self.max - self.min) * (self.upper - self.lower) + self.lower
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def inv_transform(self, data):
        if data.shape[-1] != self.max.shape[-1]:
            max = self.max[-1:]
            min = self.min[-1:]
            return max - (data - self.lower) / (self.upper - self.lower) * (max - min)
        else:
            return self.max - (data - self.lower) / (self.upper - self.lower) * (self.max - self.min)
        


def get_device(manual_input=''):
    if manual_input != '':
        return manual_input
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
        
