import torch.nn.functional as F
from torch import nn

from tsai.models.all import RNNAttention

class RNNAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__() 
        self.model = RNNAttention(c_in = 1, c_out = 1, **kwargs).float() # bs x length x channels

    def forward(self, x):
        x = x.transpose(1, 2)
        
        preds = self.model(x)
        output =  F.sigmoid(preds)
        return output