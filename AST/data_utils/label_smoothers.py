import numpy as np

class OneSmoother:
    def __init__(self, kernel_size):
        self.kernel = np.ones((kernel_size,))
    
        print('using one-smoother')
    
    def __call__(self, x):
        return np.clip(np.convolve(x, self.kernel, 'same'), 0, 1)
    
class ExponentialSmoother:
    def __init__(self, onset_decay, onset_thres):
        assert onset_decay > 0 and onset_decay < 1
        assert onset_thres > 0 and onset_thres < 1
        
        print('using exponential-smoother')
    
        if onset_decay == 0:
            kernel_side_size = 0
        else:
            kernel_side_size = int(np.floor(np.log(onset_thres) / np.log(onset_decay)))
        kernel_size = 1 + 2 * kernel_side_size
        
        kernel = np.zeros((kernel_size,))
        kernel[kernel_side_size] = 1
        for ii in range(1, kernel_side_size + 1):
            val = np.power(onset_decay, ii)
            kernel[kernel_side_size - ii] = val
            kernel[kernel_side_size + ii] = val
            
        self.kernel = kernel
    
    def __call__(self, x):
        return np.clip(np.convolve(x, self.kernel, 'same'), 0, 1)
    
class NormalSmoother:
    def __init__(self, std, size_factor=2):
        def normal_dist(x):
            return np.exp(-1/2 * np.square(x / std))
        
        print('using normal-smoother')
        
        kernel_side_size = int(np.ceil(std * size_factor))
        
        self.kernel = normal_dist(np.arange(-kernel_side_size, kernel_side_size+1))
        
    def __call__(self, x):
        return np.clip(np.convolve(x, self.kernel, 'same'), 0, 1)