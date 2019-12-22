import torch
import torch.nn as nn

from functional import log_softmax, one_hot_encoding

class LogisticRegression_torch(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.EPS = 1e-12
        
        self.linear = nn.Linear(num_features, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, X):
        out = self.logsoftmax(self.linear(X))
        return out
    

def train_torch(model, criterion, optimizer, X, y, num_itr=100, 
                num_classes=5, device=torch.device("cpu")):
    model.train()
    
    # Preprocessing
    y = one_hot_encoding(y, num_classes)
    X = torch.tensor(X)[None]
    y = torch.tensor(y)[None].contiguous()
    
    for itr in range(num_itr):
        optimizer.zero_grad()
        out = model(X.float().to(device))
        
        loss = criterion(y.to(device), out)
        loss.backward()
        optimizer.step()
        
    return model

def score_torch(model, X, y, num_classes=5, device=torch.device("cpu")):
    model.eval()
    
    # Preprocessing
    y = one_hot_encoding(y, num_classes)
    X = torch.tensor(X)[None]
    y = torch.tensor(y)[None].contiguous()
    
    out = model(X.float().to(device))
    out_prob = torch.exp(out)
    
    rights = (torch.argmax(out_prob, axis=2).cpu() == torch.argmax(y, axis=2).cpu()).sum()
    acc = float(rights) / float(X.shape[1])
        
    return acc