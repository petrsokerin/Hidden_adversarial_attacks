import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .s4_utils import S4Block, Activation


class SeqS4(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, dropout=0.2, activ_type=None):
        super().__init__()
        self.input_projector = nn.Linear(1, hidden_dim)
        self.seq_model = S4Block(
            d_model=hidden_dim,
            transposed=False,
            tie_dropout=False,
            dropout=dropout
        )

        self.outp_projector = nn.Linear(hidden_dim, output_dim)
        self.final_activ = Activation(activ_type)

    def forward(self, data, use_sigmoid=True, use_tanh=False):
        proj_data = self.input_projector(data)
        _, (hidden, _) = self.seq_model(proj_data)
        print(hidden.shape)
        hidden = hidden.reshape(hidden.shape[1], hidden.shape[2])

        return self.final_activ(self.outp_projector(hidden))