from typing import Tuple

import torch


def boltzmann(tensor: torch.Tensor, beta: float, dim: Tuple = None) -> torch.Tensor:
    exp = torch.exp(beta * tensor)
    maximum = (exp * tensor).sum(dim=dim) / exp.sum(dim=dim)
    return maximum


def boltzman_loss(
    loss_attack, reg_attack, beta: float, dim: Tuple = None
) -> torch.Tensor:
    loss_attack_exp = torch.exp(loss_attack * beta)
    reg_attack_exp = torch.exp(reg_attack * beta)
    maximum = (loss_attack_exp * loss_attack + reg_attack_exp * reg_attack) / (
        loss_attack_exp + reg_attack_exp
    )
    return maximum
