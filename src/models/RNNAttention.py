import torch
import torch.nn.functional as F
from torch import nn

from tsai.all import *

def set_requires_grad(m, requires_grad, batch_size):
    for param in m.parameters():
        param.requires_grad_(requires_grad)

class RNNA(nn.Module):
    def __init__(self, **kwargs):
        super().__init__() 
        self.model = RNNAttention(c_in = 1, c_out = 1, **kwargs).float() # bs x length x channels

        # c_in - input channels
        # c_out - output channels
        # seq_len - sequence length


    def forward(self, x):
        x = x.transpose(1, 2)
        
        preds = self.model(x)
        output =  F.sigmoid(preds)
        return output