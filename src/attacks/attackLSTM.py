import torch
from torch import nn

from .trainable_attack import Activation, TrainableAttack
from src.attacks.base_attacks import BaseIterativeAttack
from src.attacks.procedures import BatchIterativeAttack
from ..estimation import BaseEstimator


class AdversarialLSTM(nn.Module):
    def __init__(self, hidden_dim=64, x_dim=1, activation_type='identity', dropout=0.25):
        super().__init__()
        self.rnn_inp = nn.LSTM(x_dim, hidden_dim, num_layers=3, batch_first=True, dropout=dropout)
        self.act = Activation(activation_type)
        self.rnn_out = nn.LSTM(hidden_dim, x_dim, num_layers=3, batch_first=True, dropout=dropout)

    def forward(self, data):
        x, _ = self.rnn_inp(data)
        x = self.act(x)
        x, _ = self.rnn_out(x)
        return x


class LSTMAttack(BaseIterativeAttack, BatchIterativeAttack, TrainableAttack):
    def __init__(
            self,
            model: torch.nn.Module,
            criterion: torch.nn.Module,
            estimator: BaseEstimator,
            logger=None,
            eps: float = 0.03,
            n_classes=2,
            *args,
            **kwargs,
    ) -> None:
        BaseIterativeAttack.__init__(self, model=model, n_steps=1, n_classes=n_classes)
        BatchIterativeAttack.__init__(self, estimator=estimator, logger=logger, n_classes=n_classes)
        self.criterion = criterion
        self.attacker = AdversarialLSTM()
        self.eps = eps
        self.is_regularized = False
        self.n_classes = n_classes

    def get_loss(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad()
        y_pred = self.model(X)

        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            y_true = y_true.view(-1).long()

        loss = self.criterion(y_pred, y_true)
        return loss

    def get_adv_data(
            self,
            X: torch.Tensor
    ) -> torch.Tensor:
        X_adv = X.data + self.eps * torch.tanh(self.attacker(X.data))
        return X_adv

    def update_data_batch_size(self, data_size, batch_size):
        self.data_size = data_size
        self.batch_size = batch_size

    def step(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.get_adv_data(X)
