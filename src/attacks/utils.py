from typing import Tuple

import torch


def boltzmann(tensor: torch.Tensor, beta: float, dim: Tuple = None) -> torch.Tensor:
    exp = torch.exp(beta * tensor)
    maximum = (exp * tensor).sum(dim=dim) / exp.sum(dim=dim)
    return maximum
