from typing import List

import torch

from src.attacks.base_attacks import BaseIterativeAttack
from src.attacks.procedures import TrainableBatchIterativeAttack
from src.estimation import BaseEstimator
from src.utils import req_grad 


class TrainAttack(BaseIterativeAttack, TrainableBatchIterativeAttack):
    def __init__(
        self,
        model: torch.nn.Module,  # learning_target_model - модель для обучения генератора
        gen_model: torch.nn.Module,
        criterion: torch.nn.Module,
        estimator: BaseEstimator,
        logger=None,
        train_eps: float = 5,
        eps: float = 0.03,
        n_steps: int = 10,
        is_clamped: bool = False,
        n_classes = 2,
        *args,
        **kwargs,
    ) -> None:
        
        
        BaseIterativeAttack.__init__(self, model=model, n_steps=n_steps, n_classes=n_classes)
        TrainableBatchIterativeAttack.__init__(self, gen_model=gen_model, estimator=estimator, logger=logger, n_classes=n_classes)
        self.criterion = criterion
        self.is_clamped = is_clamped
        self.train_eps = train_eps
        self.eps = eps

        req_grad(self.model, False)
        req_grad(self.gen_model, True)
        self.gen_model.train()

        self.is_regularized = False
        self.n_classes = n_classes

        self.model

    def step(self, X: torch.Tensor, y_true: torch.Tensor, mode='val') -> torch.Tensor:
        if mode=='train':
            self.gen_model.train()
            delta_raw = self.gen_model(X)
            # delta_normalized = (delta_raw - delta_raw.mean(dim=0)) / (delta_raw.std(dim=0) + 1e-5)  # for l_2 norm
            delta_normalized = torch.tanh(delta_raw)  # for l_infty norm
            delta = self.train_eps * delta_normalized
            # print(torch.norm(delta), torch.norm(delta_normalized), torch.norm(X))
        else:
            self.gen_model.eval()
            delta_raw = self.gen_model(X)
            # delta_normalized = (delta_raw - delta_raw.mean(dim=0)) / (delta_raw.std(dim=0) + 1e-5)  # for l_2 norm
            # delta = self.eps * torch.tanh(delta_raw)
            delta = self.eps * torch.sign(delta_raw)  # maximize l_2 product with gradient in B_infty(eps)
            # print(torch.norm(delta))

        if self.is_clamped:
            delta = torch.clamp(delta, -1, 1)
        
        X_adv = X + delta

        return X_adv
