import torch
from tsai.models.all import RNNAttention

from .base_model import BaseModel
from .utils import Activation


class RNNA(BaseModel):
    def __init__(self, activation_type: str = "sigmoid", **kwargs) -> None:
        super().__init__()
        self.model = RNNAttention(
            c_in=1, c_out=1, **kwargs
        ).float()  # bs x length x channels
        self.final_activation = Activation(activation_type)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.transpose(1, 2)
        output = self.model(X)
        return self.final_activation(output)
