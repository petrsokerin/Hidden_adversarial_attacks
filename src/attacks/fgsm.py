from typing import List

import torch

from src.attacks.base_attacks import BaseIterativeAttack
from src.attacks.procedures import BatchIterativeAttack
from src.attacks.regularizers import reg_disc, reg_neigh
from src.attacks.utils import boltzman_loss
from src.estimation import BaseEstimator


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


class FGSMRandomAttack(FGSMAttack):
    def step(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        signs = torch.sign(torch.randn(X.shape)).to(X.device)
        X_adv = X.data + self.eps * signs
        return X_adv

class RandomAttack(FGSMAttack):
    def step(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        signs = torch.normal(0, self.eps, size=X.shape).to(X.device)
        X_adv = X.data + signs
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

class FGSMRegDiscNormAttack(FGSMAttack):
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

    def get_adv_data(self, X: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        grad_sign = torch.sign(grad)
        X_adv = X.data + self.eps * grad_sign
        return X_adv

    def step(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = self.get_loss(X, y_true)
        loss_print = loss

        reg_value = reg_disc(X, self.disc_models, self.use_sigmoid)
        # print(loss.item(), reg_value.item())

        loss_grad = torch.autograd.grad(loss, X, retain_graph=True)[0]
        reg_grad = torch.autograd.grad(reg_value, X, retain_graph=True)[0]
        loss_grad_norm = torch.norm(loss_grad, dim=0) + 1e-7
        reg_grad_norm = torch.norm(reg_grad, dim=0) + 1e-7

        loss_grad_normilized = loss_grad / loss_grad_norm
        reg_grad_normilized = reg_grad / reg_grad_norm

        # print('Class grad: ', loss_grad_norm.abs().mean().item(), 'Disc grad: ',  reg_grad_norm.abs().mean().item())
        # print('Class loss: ', loss_print.item(), 'Disc loss: ', reg_value.item())

        loss_grad = loss_grad_normilized - self.alpha * reg_grad_normilized
        X_adv = self.get_adv_data(X, loss_grad)

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

        reg_value = -reg_disc(X, self.disc_models, self.use_sigmoid)
        loss_bolzman = boltzman_loss(loss, reg_value, beta=self.beta)
        X_adv = self.get_adv_data(X, loss_bolzman)
        return X_adv


class DefenseRegDiscAttack(FGSMAttack):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        disc_models: List[torch.nn.Module],
        estimator,
        eps: float = 0.03,
        n_steps: int = 10,
        
        use_sigmoid: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model, criterion, estimator, eps, n_steps=n_steps)
        self.disc_models = disc_models
        self.use_sigmoid = use_sigmoid
        self.is_regularized = True

    def step(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        reg_value = reg_disc(X, self.disc_models, self.use_sigmoid)
        loss = - reg_value

        X_adv = self.get_adv_data(X, loss)
        return X_adv

class FGSMAttackHarmonicLoss(FGSMAttack):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        disc_models: List[torch.nn.Module],
        estimator,
        eps: float = 0.03,
        n_steps: int = 10,
        use_sigmoid: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model, criterion, estimator, eps, n_steps=n_steps)
        self.disc_models = disc_models
        self.use_sigmoid = use_sigmoid
        self.is_regularized = False


    def step(self, X: torch.Tensor, y_true: torch.Tensor, e=0.0001) -> torch.Tensor:


        loss = self.get_loss(X, y_true)
        loss_discriminator = reg_disc(X, self.disc_models, self.use_sigmoid)
        loss_harmonic_mean = 2 * (loss * loss_discriminator) / (loss + loss_discriminator + e)
        X_adv = self.get_adv_data(X, loss_harmonic_mean)
        
        return X_adv
