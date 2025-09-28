import torch
import tsai.models.all as mdls

from .base_model import BaseModel
from .utils import Activation


class PatchTST(BaseModel):
    def __init__(self, activation_type: str = "sigmoid", inference_mode=False, **kwargs) -> None:
        super().__init__()
        self.model = mdls.PatchTST(**kwargs).float()
        self.final_activation = Activation(activation_type)
        self.c_in = kwargs["c_in"]
        self.inf_mode = inference_mode

        if self.c_in > 1:
            self.pred_head = torch.nn.Sequential(
                torch.nn.Linear(self.c_in, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1)
                )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.transpose(1, 2)
        output = self.model(X)

        if self.c_in > 1:
            output = self.pred_head(output.transpose(2, 1))

        return self.final_activation(output).squeeze(-1)
