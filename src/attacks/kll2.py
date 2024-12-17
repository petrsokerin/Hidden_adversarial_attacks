from typing import List

import torch

from src.attacks.base_attacks import BaseIterativeAttack
from src.attacks.procedures import ClippedBatchIterativeAttack
from src.attacks.regularizers import reg_disc, reg_neigh
from src.attacks.utils import boltzman_loss
from src.estimation import BaseEstimator


class TrainableNoise(torch.nn.Module):
    def __init__(self, noise, model):
        super(TrainableNoise, self).__init__()
        self.noise = torch.nn.Parameter(noise)
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, X):
        return self.model(X + self.noise)


class KLL2Attack(BaseIterativeAttack, ClippedBatchIterativeAttack):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        estimator: BaseEstimator,
        logger=None,
        eps: float = 0.001,
        n_steps: int = 10,
        eta: float = 0.03,
        smoothness: float = 0.1,
        mu: float = 1.0,
        *args,
        **kwargs,
    ) -> None:

        BaseIterativeAttack.__init__(self, model=model, n_steps=n_steps)
        ClippedBatchIterativeAttack.__init__(self, estimator=estimator, logger=logger)
        self.criterion = criterion
        self.eps = eps
        self.is_regularized = False
        self.eta = eta
        self.mu = mu
        self.smoothness = smoothness

    def get_loss(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad()
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y_true)
        return loss

    def get_adv_data(
        self,
        X: torch.Tensor,
        y_true: torch.Tensor=None,
        X_orig: torch.Tensor=None,
    ) -> torch.Tensor:

        r = torch.randint(low=-1, high=1, size=X.shape) * self.eps
        trainable_r = TrainableNoise(r, self.model)

        opt = torch.optim.Adam(trainable_r.parameters())
        opt.zero_grad()

        y_pred = trainable_r(X)
        kl_loss = - self.mu * self.criterion(y_pred, y_true)

        coef_shifted = torch.ones(X.shape)
        coef_shifted[:, -1, :] = 0
        shifted_r = torch.roll(r, shifts=-1, dims=1) * coef_shifted

        fused_lasso = self.smoothness * torch.sum(torch.abs(r - shifted_r), axis=1)

        l2_loss = torch.norm(r, dim=1)

        kll2_loss = torch.mean(kl_loss + l2_loss + fused_lasso)

        kll2_loss.backward()
        opt.step()

        r = trainable_r.noise
        perturbations = X + r - X_orig

        perturbations = torch.clamp(perturbations, min=-self.eta, max=self.eta)

        X_adv = X_orig + perturbations
        return X_adv

    def step(self, X: torch.Tensor, y_true: torch.Tensor, X_orig: torch.Tensor) -> torch.Tensor:
        X_adv = self.get_adv_data(X, y_true, X_orig)
        return X_adv
