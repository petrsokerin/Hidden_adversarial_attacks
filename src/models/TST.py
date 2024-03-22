import torch.nn.functional as F
from torch import nn

import tsai.models.all as mdls

class TST(nn.Module):
    def __init__(self, **kwargs):
        super().__init__() 
        self.model = mdls.TST(c_in=1, c_out=1, **kwargs).float()

    def forward(self, x):
        x = x.transpose(1, 2)
        #x = torch.unsqueeze(x, 1).float()
        # = x.view([-1, 1, 500]).float() # 50 если slice=true
        
        preds = self.model(x)
        output =  F.sigmoid(preds)
        return output