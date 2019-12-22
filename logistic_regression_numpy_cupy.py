import numpy as np
import cupy as cp

from functional import one_hot_encoding, log_softmax

class LogisticRegression_py():
    def __init__(self, num_classes, xp=cp):
        self.num_classes = num_classes
        self.EPS = 1e-12
        self.xp = xp
    
    def fit(self, X, y, num_itr=10, lr=0.0001):
        xp = self.xp
        
        # Define the number of features
        self.num_features = X.shape[1]
        self.lr = lr
        
        # Define weights
        self.W = xp.random.rand(self.num_features, self.num_classes)

        # Train
        X = xp.array(X)
        y = one_hot_encoding(y, self.num_classes)
        y = xp.array(y, dtype=xp.int64)
        
        for itr in range(num_itr):
            loss, gradient, acc = self.train_step(X, y)

            self.W = self.W - self.lr * gradient
        
    def train_step(self, X, y):
        xp = self.xp

        # Forward
        out = X.dot(self.W)
        out_prob = log_softmax(out, xp=xp)
        
        # Accuracy
        prediction = xp.exp(out_prob)
        rights = (xp.argmax(prediction, axis=1) == xp.argmax(y, axis=1)).sum()
        acc = float(rights) / float(X.shape[0])

        # Loss
        loss = - xp.multiply( y[y > 0] , out_prob[y > 0] ).mean()

        # Gradient
        gradient =  - xp.matmul(X.T, y - out_prob)
        
        return loss, gradient, acc
    
    def score(self, X, y):
        xp = self.xp
        
        X = xp.array(X)
        y = one_hot_encoding(y, self.num_classes)
        y = xp.array(y, dtype=xp.int64)
        
        # Forward
        out = X.dot(self.W)
        out_prob = log_softmax(out, xp=xp)
        
        # Accuracy
        prediction = xp.exp(out_prob)
        rights = (xp.argmax(prediction, axis=1) == xp.argmax(y, axis=1)).sum()
        acc = float(rights) / float(X.shape[0])
        
        return acc
        