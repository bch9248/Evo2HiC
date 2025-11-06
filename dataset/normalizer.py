import numpy as np
import torch

class Normalizer:
    def __init__(self, normalization, max_reads=1e5, thres=1e-4, denominator = 100, step = None, eps=1e-5) -> None:
        self.normalization = normalization
        self.max_reads = max_reads
        self.thres=thres
        # denominator for log normalization
        self.denominator = denominator
        self.max_log = np.log(max_reads/denominator)
        # step for mixed
        self.step = step if step is not None else np.log(denominator)/(denominator-1)/self.max_log

        self.eps = eps

    def normalize(self, x, tensor = False):
        if self.normalization == 'none':
            return x
        elif self.normalization == 'mixed':
            if tensor:
                a = torch.log(x/self.denominator + self.eps)/self.max_log
                b = (x - self.denominator) * self.step
                return torch.where(x<self.denominator, b, a)
            else:
                a = np.log(x/self.denominator + self.eps)/self.max_log
                b = (x - self.denominator) * self.step
                return np.where(x<self.denominator, b, a)
        elif self.normalization == 'log1p':
            if tensor:
                return torch.log((1 + x)/self.denominator)/self.max_log
            else:
                return np.log((1 + x)/self.denominator)/self.max_log
        elif self.normalization == 'linear':
            if tensor:
                return x/self.max_reads
            else:
                return x/self.max_reads
        else:
            raise NotImplementedError

    def unnormalize(self, x, tensor = False, remove_near_zero = False):
        if self.normalization == 'none':
            x_u = x
        elif self.normalization == 'mixed':
            if tensor:
                a = (torch.exp(x*self.max_log) - self.eps) * self.denominator
                b = x/self.step + self.denominator
                return torch.where(x<0, b, a)
            else:
                a = (np.exp(x*self.max_log) - self.eps) * self.denominator
                b = x/self.step + self.denominator
                return np.where(x<0, b, a)
        elif self.normalization == 'log1p':
            if tensor:
                x_u = (torch.exp(x*self.max_log)) * self.denominator -1
            else:
                x_u = (np.exp(x*self.max_log)) * self.denominator -1
        elif self.normalization == 'linear':
            if tensor:
                x_u = x*self.max_reads
            else:    
                x_u = x*self.max_reads
        else:
            raise NotImplementedError
        
        if remove_near_zero:
            mask = x_u < self.thres
            x_u[mask] = 0

        return x_u    

