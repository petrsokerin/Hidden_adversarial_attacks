import torch
import torch.nn as nn
from utils import Activation

from .s4_utils import S4Block


class S4(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        output_dim: int = 1,
        dropout: float = 0.2,
        activation_type: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.input_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.seq_model = S4Block(
            d_model=hidden_dim, transposed=False, tie_dropout=False, dropout=dropout
        )

        self.outp_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self.final_activation = Activation(activation_type)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        proj_data = self.input_projector(X)
        output, _ = self.seq_model(proj_data)
        output = output[:, -1, :]
        return self.final_activation(output)
