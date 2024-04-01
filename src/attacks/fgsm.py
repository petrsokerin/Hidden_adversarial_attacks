import torch

from .base_attacks import BaseIterativeAttack
from .regularizers import reg_neigh, reg_disc
from .procedures import BatchIterativeAttack

class FGSMAttack(BaseIterativeAttack, BatchIterativeAttack):
    def __init__(
            self, 
            model, 
            criterion, 
            estimator,
            eps=0.03, 
            n_steps=10, 
            *args,
            **kwargs,
        ):
        BaseIterativeAttack.__init__(self, model=model, n_steps=n_steps)
        BatchIterativeAttack.__init__(self, estimator=estimator)
        self.criterion = criterion
        self.eps = eps
        self.is_regularized = False

    def get_loss(self, X, y_true):
        self.model.zero_grad()
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y_true)
        return loss
    
    def get_adv_data(self, X, loss):
        grad = torch.autograd.grad(loss, X, retain_graph=True)[0]
        grad_sign = torch.sign(grad)
        X_adv = X.data + self.eps * grad_sign
        return X_adv

    def step(self, X, y_true):
        loss = self.get_loss(X, y_true)
        X_adv = self.get_adv_data(X, loss)
        return X_adv
    

class FGSMRegNeighAttack(FGSMAttack):
    def __init__(
            self,
            model,
            criterion,
            estimator,
            eps=0.03,
            alpha=0.0,
            n_steps=10,
            *args,
            **kwargs
        ):
        super().__init__(model, criterion, estimator, eps, n_steps=n_steps)
        self.alpha = alpha
        self.is_regularized = True

    def step(self, X, y_true):
        loss = self.get_loss(X, y_true)

        reg_value = reg_neigh(X, self.alpha)
        loss = loss - reg_value

        X_adv = self.get_adv_data(X, loss)
        return X_adv


class FGSMRegDiscAttack(FGSMAttack):
    def __init__(
            self, 
            model, 
            criterion, 
            disc_models, 
            estimator,
            eps=0.03, 
            alpha=0.0, 
            n_steps=10, 
            use_sigmoid=False,
            *args,
            **kwargs,
        ):
        super().__init__(model, criterion, estimator, eps, n_steps=n_steps)
        self.alpha = alpha
        self.disc_models = disc_models
        self.use_sigmoid = use_sigmoid
        self.is_regularized = True

    def step(self, X, y_true):
        loss = self.get_loss(X, y_true)

        reg_value = reg_disc(X, self.alpha, self.disc_models, self.use_sigmoid)
        loss = loss - reg_value

        X_adv = self.get_adv_data(X, loss)
        return X_adv

