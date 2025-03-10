from typing import List

import torch

from src.attacks.base_attacks import BaseIterativeAttack
from src.attacks.procedures import KLLL2IterativeAttack
from src.attacks.regularizers import reg_disc, reg_neigh
from src.attacks.utils import boltzman_loss
from src.estimation import BaseEstimator


class TrainableNoise(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, eps: float, batch_size: int=1, low: int=-1, high: int=1, data_size=None):
        super(TrainableNoise, self).__init__()

        self.model = model
        self.device = 'cpu'
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = False
            if i == 0:
                self.device = param.device

        #self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
        self.batch_size = batch_size
        self.low = low
        self.high = high
        self.eps = eps
        if data_size is not None:
            noise = torch.randint(low=self.low, high=self.high, size=data_size) * self.eps
            self.noise = torch.nn.Parameter(noise.to(torch.float).to(self.device))
        else:
            self.noise = None

    def init_noise(self, data_size: int, batch_size: int, reinit=False):
        self.batch_size = batch_size
        if self.noise is None or reinit:
            noise = torch.randint(low=-1, high=1, size=data_size) * self.eps
            self.noise = torch.nn.Parameter(noise.to(torch.float).to(self.device))

    def forward(self, X: torch.Tensor, batch_id: int):
        data_noise = self.noise[self.batch_size * (batch_id): self.batch_size * (batch_id + 1)]
        # print(self.batch_size * (batch_id), self.batch_size * (batch_id + 1))
        # print(batch_id, data_noise.shape, X.shape)
        return self.model(X + data_noise), data_noise


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
        self.eps = eps
        self.mu = mu
        self.smoothness = smoothness

        self.device = 'cpu'
        for param in self.model.parameters():
            self.device = param.device
            break

        if norm_coef is None:
            self.norm_coef = smoothness
        else:
            self.norm_coef = norm_coef

        self.trainable_r = TrainableNoise(self.model, self.eps)

    def kll2_loss(self, X:torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, r: float):
        kl_loss = self.mu * self.criterion(y_pred, y_true)

        coef_shifted = torch.ones(X.shape).to(r.device)
        coef_shifted[:, -1, :] = 0
        shifted_r = torch.roll(r, shifts=-1, dims=1) * coef_shifted

        fused_lasso = self.smoothness * torch.sum(torch.abs(r - shifted_r), axis=1)

        l2_loss = self.norm_coef * torch.norm(r, dim=1)

        kll2_loss = torch.mean(- kl_loss + l2_loss + fused_lasso)
        return kll2_loss
    
    def reinit_attack(self):
        #print("Data, batch size", self.data_size, self.batch_size)
        self.trainable_r.init_noise(self.data_size, self.batch_size, reinit=True)

    def update_data_batch_size(self, data_size: int, batch_size: int):
        self.data_size = data_size
        self.batch_size = batch_size

    def get_adv_data(
        self,
        X_orig: torch.Tensor = None,
        y_true: torch.Tensor = None,
        batch_id: int = 0,
    ) -> torch.Tensor:

        self.trainable_r.init_noise(self.data_size, self.batch_size)
        self.opt = torch.optim.Adam(self.trainable_r.parameters(), lr=0.01)
        self.opt.zero_grad()

        y_pred, noise = self.trainable_r(X_orig, batch_id)
        kll2_loss = self.kll2_loss(X_orig, y_true, y_pred, noise)

        kll2_loss.backward()
        self.opt.step()

        perturbations = self.trainable_r.noise[self.batch_size*(batch_id): self.batch_size*(batch_id+1)]
        perturbations = torch.clamp(perturbations, min=-self.eta, max=self.eta)

        X_adv = X_orig + perturbations
        return X_adv

    def step(self, X: torch.Tensor, y_true: torch.Tensor, X_orig: torch.Tensor, batch_id: int) -> torch.Tensor:
        X_adv = self.get_adv_data(X_orig, y_true, batch_id)
        return X_adv
