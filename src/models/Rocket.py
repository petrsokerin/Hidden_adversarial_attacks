import torch
import tsai.models.all as mdls

from .base_model import BaseModel
from .utils import Activation


class Rocket(BaseModel):
    def __init__(self, activation_type: str = "sigmoid", **kwargs) -> None:
        super().__init__()
        self.model = mdls.ROCKET(**kwargs).float().to(torch.device(kwargs['device']))
        
        self.head_device = torch.device(kwargs['device'])
        
        self.classifier = torch.nn.Linear(2*self.model.n_kernels, 1)

        self.final_activation = Activation(activation_type)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.transpose(1, 2)
        output = self.model(X).to(self.head_device)
        output = self.classifier(output)
        return self.final_activation(output)
