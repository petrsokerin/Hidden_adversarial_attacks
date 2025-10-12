from typing import Optional
import torch
from torch import nn
import tsai.models.all as mdls

from .base_model import BaseModel
from .utils import Activation


class AttackPatchTST(BaseModel):
    """
    PatchTST-based surrogate model for generating adversarial perturbations.
    
    Architecture: Input -> PatchTST -> Activation -> Linear -> Output
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        x_dim: int = 1,
        activation_type: str = "tanh",
        patch_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.x_dim = x_dim
        self.activation_type = activation_type
        
        # PatchTST backbone
        patch_kwargs = patch_kwargs or {}
        # Ensure correct output dimensions for PatchTST
        patch_kwargs.update(dict(
            c_in=x_dim,
            c_out=hidden_dim,
            pred_dim=hidden_dim
        ))
        
        self.step_model = mdls.PatchTST(**patch_kwargs)
        self.fc = nn.Linear(hidden_dim, x_dim)
        self.act = Activation(activation_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, x_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, x_dim)
        """
        B, L, _ = x.shape
        # PatchTST expects (B, C, L) format
        # x_transposed = x.transpose(1, 2)  # (B, x_dim, L)
        
        # Get hidden representation: (B, hidden_dim, 1)
        h = self.step_model(x)
        
        # Reshape: (B, hidden_dim, 1) -> (B, hidden_dim)
        h = h.view(B, -1)
        # Expand to sequence length: (B, hidden_dim) -> (B, L, hidden_dim)
        #! CONST SHIFT PROBLEM
        h = h.unsqueeze(1).expand(-1, L, -1)
        return self.fc(self.act(h))
