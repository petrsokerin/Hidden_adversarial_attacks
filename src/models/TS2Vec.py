import torch
import torch.nn.functional as F

from src.models.TS2Vec_src.ts2vec import TS2Vec

from .base_model import BaseSelfSupervisedModel
from .utils import Activation, build_head


class TS2VEC(BaseSelfSupervisedModel):
    def __init__(
        self,
        emb_size: int = 320,
        input_dim: int = 1,
        n_layers: int = 3,
        n_classes: int = 2,
        emb_batch_size: int = 16,
        fine_tune: bool = False,
        dropout: float = 0.0,
        dropout_ts2vec: float = 0.0,
        device: str = "cuda:0",
        activation_type: str = "sigmoid",
    ) -> None:
        super().__init__()

        if n_classes == 2:
            output_size = 1
        else:
            output_size = n_classes
        self.device = device
        self.ts2vec = TS2Vec(
            input_dims=input_dim,
            dropout=dropout_ts2vec,
            device=device,
            output_dims=emb_size,
            batch_size=emb_batch_size,
        )

        self.emd_model = self.ts2vec.net
        if not fine_tune:
            for param in self.emd_model.parameters():
                param.requires_grad = False

        self.classifier = build_head(
            emb_size, output_size, n_layers=n_layers, dropout=dropout
        )
        self.final_activation = Activation(activation_type)

    def train_embedding(self, X_train: torch.Tensor, verbose=True) -> None:
        self.ts2vec.fit(X_train, verbose=verbose)
        self.emd_model = self.ts2vec.net

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        emb_out = self.emd_model(X, mask)
        emb_out = F.max_pool1d(emb_out.transpose(1, 2), kernel_size=emb_out.size(1))
        emb_out = emb_out.transpose(1, 2).squeeze(1)
        out = self.classifier(emb_out)
        return self.final_activation(out)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
