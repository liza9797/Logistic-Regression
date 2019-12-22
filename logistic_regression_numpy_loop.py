import numpy as np

from functional import log_softmax_loop, matrix_multiplication_loop

def forward_step_loop(X, y, W):
    
    # Matrix multiplication
    out = matrix_multiplication_loop(X, W)
            
    # LogSoftMax
    out = log_softmax_loop(out)
    
    # Loss
    loss = 0.
    for i in range(y.shape[0]):
        loss -= out[i, y[i]]
    loss = loss / y.shape[0]
    
    # Gradient
    gradient = np.zeros_like(W)
    
    for i in range(gradient.shape[0]): # over 13
        for j in range(gradient.shape[1]): # over 5
            for k in range(y.shape[0]): # over 303
                gradient[i, j] += X[k, i] * out[k, j]
                if y[k] == j:
                    gradient[i, j] -= X[k, i] 

    # Accuracy
    acc = calulate_accuracy(out, y)
    
    return loss, gradient, acc

def forward_loop(X, y, W=None, num_itr=100, num_classes=5, lr=1e-4, show_train=True):
    num_features = X.shape[1]
    
    if not W:
        W = np.random.rand(num_features, num_classes)
    
    for itr in range(num_itr):
        loss, gradient, acc = forward_step_loop(X, y, W)
        if loss > 1e10:
            break
        if show_train:
            print("{} itr: loss={}, acc={}".format(itr+1, loss, acc))
        W = W - lr * gradient
        
    return W

def calulate_accuracy(outputs, y):
    
    # Prediction
    predictions = []
    for i in range(outputs.shape[0]):
        max_prob = 0.
        for j in range(outputs.shape[1]):
            pred = 0
            prob = np.exp(outputs[i, j])
            if prob >= max_prob:
                max_prob = prob
                pred = j
                
        predictions.append(pred)
    predictions = np.array(predictions)
    
    # Accuracy
    acc = (predictions == y).sum() / predictions.shape[0]
    return acc
    
    
def score_loop(X, y, W):
    # Matrix multiplication
    out = matrix_multiplication_loop(X, W)
            
    # LogSoftMax
    out = log_softmax_loop(out)
    
    # Accuracy
    acc = calulate_accuracy(out, y)
    return acc