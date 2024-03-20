from .utils import req_grad
import torch
from typing import List


def reg_neigh(x, alpha):
    x_anchor = x[:, 1:-1]
    x_left = x[:, 2:]
    x_right = x[:, :-2]
    x_regular = (x_left + x_right) / 2
    reg_value = torch.sum((x_anchor - x_regular.detach()) ** 2, dim=list(range(1, len(x.shape))))
    reg_value = alpha * torch.mean(reg_value)
    return reg_value


def reg_disc(x, alpha: float, disc_models: List, use_sigmoid: bool = True):
    n_models = len(disc_models)
    reg_value = 0
    for d_model in disc_models:
        req_grad(d_model, state=True)
        if use_sigmoid:
            model_output = torch.mean(torch.log(F.sigmoid(d_model(x))))
        else:
            model_output = torch.mean(torch.log(d_model(x)))
        reg_value = reg_value + model_output

    reg_value = alpha * reg_value / n_models
    return reg_value


def reg_boltzmann(x, alpha: float, disc_models: List, use_sigmoid: bool = True):
    def boltzmann(tensor, alpha, dim=None):  # alpha = 85 ~ torch.max(dim=0)
        exp = torch.exp(alpha * tensor)
        maximum = (exp * tensor).sum(dim=dim) / exp.sum(dim=dim)

        return maximum

    reg_value = torch.empty(len(disc_models))

    for i, d_model in enumerate(disc_models):
        req_grad(d_model, state=True)
        if use_sigmoid:
            model_output = torch.mean(torch.log(F.sigmoid(d_model(x))))
        else:
            model_output = torch.mean(torch.log(d_model(x)))

        reg_value[i] = model_output

    # print('bolt: ', boltzmann(reg_value, alpha=10))
    # print('max: ', reg_value.max())

    reg_value = alpha * boltzmann(reg_value, alpha=30)
    # reg_value = alpha* reg_value.max()
    return reg_value