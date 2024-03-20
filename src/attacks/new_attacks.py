import torch
import numpy as np


def fgsm_clip_attack(
        model,
        loss_func,
        x,
        y_true,
        eps: float,
        n_steps: int = 20,
        scale_factor: float = 1,
        max_weight: float = 1,
):
    # w(x) = exp(-(x/scale_factor)^2) * max_weight
    weight_func = lambda val: np.exp(-(val / scale_factor)**2) * max_weight
    x_new = x
    for i in range(n_steps):
        y_pred = model(x_new)
        loss_val = loss_func(y_pred, y_true)
        grad = torch.autograd.grad(loss_val, x_new, retain_graph=True)[0]
        x_temp = x.data + eps * torch.sign(grad).data

        clip_val = weight_func(torch.dist(x, x_temp))
        x_new = x.data + eps * torch.clip(grad, -clip_val, clip_val).data

    return x_new