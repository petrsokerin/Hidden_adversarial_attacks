from typing import List

import torch

from src.attacks.base_attacks import BaseIterativeAttack
from src.attacks.procedures import KLLL2IterativeAttack
from src.attacks.regularizers import reg_disc, reg_neigh
from src.attacks.utils import boltzman_loss
from src.estimation import BaseEstimator


class TrainableNoise(torch.nn.Module):
    def __init__(self, model, eps, batch_size=1, low=-1, high=-1, data_size=None):
        super(TrainableNoise, self).__init__()

        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        self.batch_size = batch_size
        self.low = low
        self.high = high
        self.eps = eps
        if data_size is not None:
            noise = torch.randint(low=self.low, high=self.high, size=data_size).unsqueeze(-1) * self.eps
            self.noise = torch.nn.Parameter(noise.to(torch.float))
        else:
            self.noise = None

    def init_noise(self, data_size, batch_size, reinit=False):
        self.batch_size = batch_size
        if self.noise is None or reinit:
            noise = torch.randint(low=-1, high=1, size=data_size).unsqueeze(-1) * self.eps
            self.noise = torch.nn.Parameter(noise.to(torch.float))

    def forward(self, X, batch_id):
        data_noise = self.noise[self.batch_size * (batch_id):self.batch_size * (batch_id + 1)]
        return self.model(X + data_noise)


class KLL2Attack(BaseIterativeAttack, KLLL2IterativeAttack):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        estimator: BaseEstimator,
        logger=None,
        eps: float = 0.001,
        n_steps: int = 10,
        eta: float = 0.03,
        mu: float = 1.0,
        smoothness: float = 0.1,
        norm_coef: float = None,
        *args,
        **kwargs,
    ) -> None:

        BaseIterativeAttack.__init__(self, model=model, n_steps=n_steps)
        KLLL2IterativeAttack.__init__(self, estimator=estimator, logger=logger)
        self.criterion = criterion
        self.is_regularized = False
        self.eta = eta
        self.eps = 2.5 * self.eta / n_steps
        self.mu = mu
        self.smoothness = smoothness
        if norm_coef is None:
            self.norm_coef = smoothness
        else:
            self.norm_coef = norm_coef

        print(self.smoothness, self.norm_coef, self.eta)

        self.trainable_r = TrainableNoise(self.model, self.eps)

    def kll2_loss(self, y_true, y_pred, r):
        kl_loss = self.mu * self.criterion(y_pred, y_true)

        coef_shifted = torch.ones(X.shape)
        coef_shifted[:, -1, :] = 0
        shifted_r = torch.roll(r, shifts=-1, dims=1) * coef_shifted

        fused_lasso = self.smoothness * torch.sum(torch.abs(r - shifted_r), axis=1)

        l2_loss = self.norm_coef * torch.norm(r, dim=1)

        kll2_loss = torch.mean(- kl_loss + l2_loss + fused_lasso)
        return kll2_loss


    def get_adv_data(
        self,
        X: torch.Tensor,
        y_true: torch.Tensor = None,
        X_orig: torch.Tensor = None,
        batch_id: int = 0,
    ) -> torch.Tensor:

        self.trainable_r.init_noise(self.data_size, self.batch_size)
        self.opt = torch.optim.Adam(self.trainable_r.parameters())
        self.opt.zero_grad()

        r = self.trainable_r.noise[self.batch_size*(batch_id): self.batch_size*(batch_id+1)]
        y_pred = self.trainable_r(X_orig, batch_id)
        kll2_loss = self.kll2_loss(self, y_true, y_pred, r)

        kll2_loss.backward()
        self.opt.step()

        perturbations = self.trainable_r.noise[self.batch_size*(batch_id): self.batch_size*(batch_id+1)]
        perturbations = torch.clamp(perturbations, min=-self.eta, max=self.eta)

        X_adv = X_orig + perturbations
        return X_adv

    def step(self, X: torch.Tensor, y_true: torch.Tensor, X_orig: torch.Tensor, batch_id: int) -> torch.Tensor:
        X_adv = self.get_adv_data(X, y_true, X_orig, batch_id)
        return X_adv
