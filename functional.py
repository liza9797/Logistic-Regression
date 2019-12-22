import numpy as np
import cupy as cp

def log_softmax(x, xp=cp):
    
    output = xp.subtract(x, x.max(axis=1, keepdims=True))
    
    coeff = xp.log( (xp.exp(x)).sum(axis=1, keepdims=True) )
    output = x - coeff
    return output

def one_hot_encoding(y, num_classes):
    
    y_out = np.zeros((y.shape[0], num_classes))
    for i in range(y.shape[0]):
        y_c = int(y[i])
        y_out[i, y_c] = 1
        
    return y_out

def matrix_multiplication_loop(X, W):
    
    out = np.zeros((X.shape[0], W.shape[1]))
    for i in range(X.shape[0]):
        for j in range(W.shape[1]):
            
            value = 0.
            for k in range(X.shape[1]):
                value += X[i, k] * W[k, j]
                
            out[i, j] = value
    return out

def log_softmax_loop(x):
    for i in range(x.shape[0]):
        sum_ = 0.
        for j in range(x.shape[1]):
            sum_ += np.exp(x[i, j])
        
        for j in range(x.shape[1]):
            x[i, j] =  x[i, j] - np.log(sum_)
    return x 

