import torch
import torch.nn as nn

from .base_model import BaseModel
from .s4_utils import S4Block
from .utils import Activation


class S4(BaseModel):
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
        output = self.outp_projector(output)
        return self.final_activation(output)


class GenS4(BaseModel):
    def __init__(
        self,
        c_in: int = 1,
        c_out: int = None,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        activation_type: str = "linear",
    ) -> None:
        super().__init__()
        c_out = c_in if c_out is None else c_out

        self.input_projector = nn.Sequential(
            nn.Linear(c_in, hidden_dim // 2),
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
            nn.Linear(hidden_dim // 2, c_out),
        )
        self.final_activation = Activation(activation_type)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, L, C_in)
        h = self.input_projector(X)      # (B, L, H)
        h, _ = self.seq_model(h)         # (B, L, H)
        out = self.outp_projector(h)     # (B, L, C_out)
        return self.final_activation(out)
