import torch.nn as nn

from .s4_utils import Activation, S4Block


class S4(nn.Module):
    def __init__(
        self, input_dim=1, hidden_dim=32, output_dim=1, dropout=0.2, activ_type=None
    ):
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
        self.final_activ = Activation(activ_type)

    def forward(self, data):
        proj_data = self.input_projector(data)
        output, _ = self.seq_model(proj_data)
        output = output[:, -1, :]
        return self.final_activ(self.outp_projector(output))
