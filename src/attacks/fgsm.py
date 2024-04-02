from typing import List

import torch

from src.estimation import BaseEstimator
from .utils import boltzman_loss

from .base_attacks import BaseIterativeAttack
from .procedures import BatchIterativeAttack
from .regularizers import reg_disc, reg_neigh

class FGSMAttack(BaseIterativeAttack, BatchIterativeAttack):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        estimator: BaseEstimator,
        eps: float = 0.03,
        n_steps: int = 10,
        *args,
        **kwargs,
    ) -> None:
        BaseIterativeAttack.__init__(self, model=model, n_steps=n_steps)
        BatchIterativeAttack.__init__(self, estimator=estimator)
        self.criterion = criterion
        self.eps = eps
        self.is_regularized = False

    def get_loss(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad()
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y_true)
        return loss

    def get_adv_data(self, X: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        grad = torch.autograd.grad(loss, X, retain_graph=True)[0]
        grad_sign = torch.sign(grad)
        X_adv = X.data + self.eps * grad_sign
        return X_adv

    def step(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = self.get_loss(X, y_true)
        X_adv = self.get_adv_data(X, loss)
        return X_adv


class FGSMRegNeighAttack(FGSMAttack):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        estimator: BaseEstimator,
        eps: float = 0.03,
        alpha: float = 0.0,
        n_steps: float = 10,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model, criterion, estimator, eps, n_steps=n_steps)
        self.alpha = alpha
        self.is_regularized = True

    def step(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = self.get_loss(X, y_true)

        reg_value = reg_neigh(X, self.alpha)
        loss = loss - reg_value

        X_adv = self.get_adv_data(X, loss)
        return X_adv


class FGSMRegDiscAttack(FGSMAttack):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        disc_models: List[torch.nn.Module],
        estimator,
        eps: float = 0.03,
        alpha: float = 0.0,
        n_steps: int = 10,
        use_sigmoid: bool = False,
        *args,
        **kwargs,
     ) -> None:
        super().__init__(model, criterion, estimator, eps, n_steps=n_steps)
        self.alpha = alpha
        self.disc_models = disc_models
        self.use_sigmoid = use_sigmoid
        self.is_regularized = True

    def step(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = self.get_loss(X, y_true)

        reg_value = reg_disc(X, self.disc_models, self.use_sigmoid)
        loss = loss - self.alpha * reg_value

        X_adv = self.get_adv_data(X, loss)
        return X_adv
    
class FGSMRegDiscSmoothMaxAttack(FGSMAttack):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        disc_models: List[torch.nn.Module],
        estimator,
        eps: float = 0.03,
        beta: float = 0.0,
        n_steps: int = 10,
        use_sigmoid: bool = False,
        *args,
        **kwargs,
     ) -> None:
        super().__init__(model, criterion, estimator, eps, n_steps=n_steps)
        self.beta = beta
        self.disc_models = disc_models
        self.use_sigmoid = use_sigmoid
        self.is_regularized = True

    def step(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = self.get_loss(X, y_true)

        reg_value = reg_disc(X, self.disc_models, self.use_sigmoid)
        loss = boltzman_loss(loss, reg_value, beta=self.beta)

        X_adv = self.get_adv_data(X, loss)
        return X_adv
