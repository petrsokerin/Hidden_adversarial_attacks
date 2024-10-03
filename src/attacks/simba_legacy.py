import torch

from .base_attacks import BaseIterativeAttack
from .regularizers import reg_disc, reg_neigh


class SimBABinary(BaseIterativeAttack):
    def __init__(self, model, criterion, eps, n_steps, device="cpu"):
        super().__init__(model, criterion, eps, n_steps, device=device)

    def step(self, x, y_true):
        rand_ind = torch.randint(x.shape[1], (1,))[0]
        mask = torch.zeros(x.shape).to(self.device)
        mask[:, rand_ind] = mask[:, rand_ind] + 1

        x_minus = x - mask * self.eps
        x_plus = x + mask * self.eps

        y_pred = self.model(x)
        y_pred_minus = self.model(x_minus)
        y_pred_plus = self.model(x_plus)

        p_orig = torch.sigmoid(y_pred)
        p_minus = torch.sigmoid(y_pred_minus)
        p_plus = torch.sigmoid(y_pred_plus)

        x_all = torch.cat([x, x_minus, x_plus]).view(3, *x.shape)
        losses = torch.stack(
            [
                self.criterion(p_orig, y_true),
                self.criterion(p_minus, y_true),
                self.criterion(p_plus, y_true),
            ],
            dim=0,
        )

        max_loss_indices = torch.argmax(losses, dim=0)
        max_loss_indices = (
            min_loss_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, x.shape[-1])
        )

        x_adv = torch.gather(x_all, 0, max_loss_indices).squeeze(0)
        return x_adv


class SimBABinaryReg(SimBABinary):
    def __init__(self, model, criterion, eps, n_steps, alpha, device="cpu"):
        super().__init__(model, criterion, eps, n_steps, device=device)
        self.alpha = alpha

    def step(self, x, y_true):
        x_adv = super().step(x, y_true)
        reg_value = reg_neigh(x_adv, self.alpha)

        loss = -reg_value
        grad_ = torch.autograd.grad(loss, x_adv, retain_graph=True)[0]
        x_adv = x_adv.data + grad_

        return x_adv


class SimBABinaryDiscReg(SimBABinary):
    def __init__(
        self, model, criterion, eps, n_steps, alpha, disc_models, device="cpu"
    ):
        super().__init__(model, criterion, eps, n_steps, device=device)
        self.alpha = alpha
        self.disc_models = disc_models

    def step(self, x, y_true):
        x_adv = super().step(x, y_true)
        reg_value = reg_disc(x_adv, self.alpha, self.disc_models)

        loss = -reg_value
        grad_ = torch.autograd.grad(loss, x_adv, retain_graph=True)[0]
        x_adv = x_adv.data + grad_

        return x_adv
