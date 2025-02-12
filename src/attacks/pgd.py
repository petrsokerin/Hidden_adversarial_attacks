from typing import List

import torch

from src.attacks.base_attacks import BaseIterativeAttack
from src.attacks.procedures import ClippedBatchIterativeAttack
from src.attacks.regularizers import reg_disc, reg_neigh
from src.estimation import BaseEstimator


class PGDAttack(BaseIterativeAttack, ClippedBatchIterativeAttack):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        estimator: BaseEstimator,
        n_steps: int = 10,
        eta: float = 0.5,
        norm: bool=None,
        logger=None,
        *args,
        **kwargs,
    ) -> None:
        BaseIterativeAttack.__init__(self, model=model, n_steps=n_steps)
        ClippedBatchIterativeAttack.__init__(self, estimator=estimator, logger=logger)
        self.criterion = criterion
        self.eta = eta
        self.eps = self.eta*2.5/self.n_steps
        self.norm = norm
        self.is_regularized = False

    def get_loss(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad()
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y_true)
        return loss

    def get_adv_data(self, X: torch.Tensor, loss: torch.Tensor, X_orig: torch.Tensor) -> torch.Tensor:

        grad = torch.autograd.grad(loss, X, retain_graph=True)[0]
        X_adv = X + self.eps * grad.sign()
        perturbation = X_adv - X_orig

        if self.norm:
            scaling_factor = self.eta / torch.norm(X_adv - X_orig, p=self.norm, dim=1).unsqueeze(-1)
            perturbation = perturbation * torch.where(scaling_factor < 1, scaling_factor, 1)
        else:
            perturbation = torch.clamp(X_adv - X_orig, min=-self.eta, max=self.eta)

        X_adv = X_orig + perturbation

        return X_adv

    def step(self, X: torch.Tensor, y_true: torch.Tensor, X_orig: torch.Tensor) -> torch.Tensor:
        loss = self.get_loss(X, y_true)
        X_adv = self.get_adv_data(X, loss, X_orig)
        return X_adv

class PGDRegDiscAttack(PGDAttack):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        disc_models: List[torch.nn.Module],
        estimator: BaseEstimator,
        alpha: float = 0.0,
        n_steps: int = 10,
        eta = 0.5,
        norm=None,
        use_sigmoid: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model, criterion, estimator, n_steps, eta, norm)
        self.alpha = alpha
        self.disc_models = disc_models
        self.use_sigmoid = use_sigmoid
        self.is_regularized = True

    def step(self, X: torch.Tensor, y_true: torch.Tensor, X_orig) -> torch.Tensor:
        loss = self.get_loss(X, y_true)

        reg_value = reg_disc(X, self.disc_models, self.use_sigmoid)
        loss = loss - self.alpha * reg_value

        X_adv = self.get_adv_data(X, loss, X_orig)
        return X_adv
