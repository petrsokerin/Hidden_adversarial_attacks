import torch
from torch import nn
from tsai.models.all import RNNAttention

from .base_model import BaseModel
from .utils import Activation


class RNNA(BaseModel):
    def __init__(self, activation_type: str = "sigmoid", **kwargs) -> None:
        super().__init__()
        self.model = RNNAttention(**kwargs).float()  # bs x length x channels
        self.final_activation = Activation(activation_type)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.transpose(1, 2)
        output = self.model(X)
        return self.final_activation(output)


class _RNNASeqHead(nn.Module):
    def __init__(self, d_model: int, c_out: int, seq_len: int):
        super().__init__()
        self.proj = nn.Conv1d(d_model, c_out, kernel_size=1)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, d_model, L)
        return self.proj(z).transpose(1, 2)  # (B, L, c_out)


class GenRNNA(BaseModel):
    def __init__(self, activation_type: str = "tanh", **kwargs) -> None:
        super().__init__()

        c_in = kwargs.get("c_in", 1)
        kwargs["c_out"] = kwargs.get("c_out", c_in)  # input length = output length
        kwargs["custom_head"] = kwargs.get("custom_head", _RNNASeqHead)
        self.model = RNNAttention(**kwargs).float()
        self.final_activation = Activation(activation_type)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, L, C)
        X = X.transpose(1, 2)              # (B, C, L) - format for RNNAttention
        output = self.model(X)             # (B, L, C)
        return self.final_activation(output)
