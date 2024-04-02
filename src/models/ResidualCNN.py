import torch
from torch import nn
from tsai.models.all import ResCNN

from .utils import Activation


class ResidualCNN(nn.Module):
    def __init__(self, activation_type: str = "sigmoid", **kwargs) -> None:
        super().__init__()
        self.model = ResCNN(c_in=1, c_out=1, **kwargs).float()
        self.final_activation = Activation(activation_type)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.transpose(1, 2)
        output = self.model(X)
        return self.final_activation(output)
