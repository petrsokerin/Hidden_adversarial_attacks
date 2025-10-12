import torch
from torch import nn

from .base_model import BaseModel
from .utils import Activation


class AttackLSTM(BaseModel):
    """
    LSTM-based surrogate model for generating adversarial perturbations.
    
    Architecture: Input -> LSTM -> Activation -> LSTM -> Output
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        x_dim: int = 1,
        activation_type: str = 'identity',
        dropout: float = 0.25,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.x_dim = x_dim
        self.activation_type = activation_type
        self.dropout = dropout
        self.num_layers = num_layers
        
        # Input LSTM: x_dim -> hidden_dim
        self.rnn_inp = nn.LSTM(
            x_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Activation layer
        self.act = Activation(activation_type)
        
        # Output LSTM: hidden_dim -> x_dim
        self.rnn_out = nn.LSTM(
            hidden_dim, 
            x_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, data):
        x, _ = self.rnn_inp(data)
        x = self.act(x)
        x, _ = self.rnn_out(x)
        return x
