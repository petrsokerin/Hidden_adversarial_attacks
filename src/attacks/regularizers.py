from typing import List

import torch
import torch.nn.functional as F

from src.attacks.utils import boltzmann
from src.utils import req_grad


def reg_neigh(X: torch.Tensor, alpha: float) -> torch.Tensor:
    X_anchor = X[:, 1:-1]
    X_left = X[:, 2:]
    X_right = X[:, :-2]
    X_regular = (X_left + X_right) / 2
    reg_value = torch.sum(
        (X_anchor - X_regular.detach()) ** 2, dim=list(range(1, len(X.shape)))
    )
    reg_value = alpha * torch.mean(reg_value)
    return reg_value


def reg_disc(
    X: torch.Tensor,
    disc_models: List[torch.nn.Module],
    use_sigmoid: bool = True,
) -> torch.Tensor:
    n_models = len(disc_models)
    reg_value = 0
    for d_model in disc_models:
        req_grad(d_model, state=True)
        if use_sigmoid:
            model_output = torch.mean(torch.log(F.sigmoid(d_model(X))))
        else:
            model_output = torch.mean(torch.log(d_model(X)))
        reg_value = reg_value + model_output

    reg_value = reg_value / n_models
    return reg_value


def reg_boltzmann(
    X: torch.Tensor, beta: float, disc_models: List, use_sigmoid: bool = True
) -> torch.Tensor:
    reg_value = torch.empty(len(disc_models))

    for i, d_model in enumerate(disc_models):
        req_grad(d_model, state=True)
        if use_sigmoid:
            model_output = torch.mean(torch.log(F.sigmoid(d_model(X))))
        else:
            model_output = torch.mean()

        reg_value[i] = model_output

    reg_value = beta * boltzmann(reg_value, beta=beta)
    return reg_value

# def reg_disc_loss(self, X: torch.Tensor,
#                 y_true: torch.Tensor,
#                 discriminator: torch.nn.Module, 
#                 discriminator_criterion: torch.nn.Module ) -> torch.Tensor:
#     discriminator.zero_grad()
#     y_pred_disc = discriminator(X)
#     disc_loss = disc_criterion(y_pred_disc, y_true)
#     return disc_loss