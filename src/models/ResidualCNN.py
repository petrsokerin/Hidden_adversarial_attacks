import torch
import torch.nn.functional as F
from torch import nn

from tsai.all import *

def set_requires_grad(m, requires_grad, batch_size):
    for param in m.parameters():
        param.requires_grad_(requires_grad)

class ResidualCNN(nn.Module):
    def __init__(self):
        super().__init__() 
        self.model = ResCNN(1, 1).float()


    def forward(self, x):
        x = x.transpose(1, 2)
        #x = torch.unsqueeze(x, 1).float()
        # = x.view([-1, 1, 500]).float() # 50 если slice=true
        
        preds = self.model(x)
        output =  F.sigmoid(preds)
        return output