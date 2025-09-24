from torch import nn


def build_head(
    emb_size: int, out_size: int, n_layers: int = 3, dropout: float = 0.0
) -> nn.Module:
    if n_layers not in [1, 2, 3]:
        raise ValueError("n layers should be in [1, 2, 3]")
    if n_layers == 3:
        return nn.Sequential(
            nn.Linear(emb_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.Linear(32, out_size),
        )
    elif n_layers == 2:
        return nn.Sequential(
            nn.Linear(emb_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, out_size),
        )
    elif n_layers == 1:
        return nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(emb_size),
            nn.Dropout(dropout),
            nn.Linear(emb_size, out_size),
        )


def Activation(activation=None, dim=-1):
    if activation in [None, "id", "identity", "linear"]:
        return nn.Identity()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "elu":
        return nn.ELU()
    elif activation in ["swish", "silu"]:
        return nn.SiLU()
    elif activation == "glu":
        return nn.GLU(dim=dim)
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softplus":
        return nn.Softplus()
    else:
        raise NotImplementedError(
            "hidden activation '{}' is not implemented".format(activation)
        )
