import os
import shutil
from typing import Dict, Iterable, List

import torch

from src import attacks, estimation, models


def get_estimator(
    estimator_name: str, estimator_params: Dict
) -> estimation.BaseEstimator:
    if estimator_params is None:
        estimator_params = dict()
    try:
        return getattr(estimation, estimator_name)(**estimator_params)
    except AttributeError:
        raise ValueError(f"Estimator with name {estimator_name} is not implemented")


def get_attack(attack_name: str, attack_params: Dict) -> attacks.BaseIterativeAttack:
    if attack_params is None:
        attack_params = dict()
    try:
        return getattr(attacks, attack_name)(**attack_params)
    except AttributeError:
        raise ValueError(f"Attack with name {attack_name} is not implemented")


def get_model(
    model_name: str,
    model_params: Dict,
    device: str = "cpu",
    path: str = None,
    train_mode: bool = False,
) -> torch.nn.Module:
    if model_params is None:
        model_params = dict()
    try:
        model = getattr(models, model_name)(**model_params)
        model = model.to(device)
        if path:
            model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        model.train(train_mode)
        return model
    except AttributeError:
        raise ValueError(f"Model with name {model_name} is not implemented")


def get_criterion(
    criterion_name: str, criterion_params: Dict = None
) -> torch.nn.Module:
    if criterion_params is None:
        criterion_params = dict()
    try:
        return getattr(torch.nn, criterion_name)(**criterion_params)
    except AttributeError:
        raise ValueError(f"Criterion with name {criterion_name} is not implemented")


def get_optimizer(
    optimizer_name: str, model_params: Dict, optimizer_params: Iterable = None
):
    if optimizer_params is None:
        optimizer_params = dict()
    try:
        return getattr(torch.optim, optimizer_name)(model_params, **optimizer_params)
    except AttributeError:
        raise ValueError(f"Optimizer with name {optimizer_name} is not implemented")


def get_scheduler(
    scheduler_name: str, optimizer: torch.optim.Optimizer, scheduler_params: Dict = None
) -> torch.optim.lr_scheduler.LRScheduler:
    if scheduler_params is None:
        scheduler_params = dict()
    try:
        return getattr(torch.optim.lr_scheduler, scheduler_name)(
            optimizer, **scheduler_params
        )
    except AttributeError:
        raise ValueError(f"Optimizer with name {scheduler_name} is not implemented")


def save_config(path: str, config_load_name: str, config_save_name: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)

    shutil.copyfile(
        f"config/{config_load_name}.yaml", path + f"/{config_save_name}.yaml"
    )


def get_disc_list(
    model_name: str,
    model_params: Dict,
    list_disc_params: List[Dict],
    device: str = "cpu",
    path: str = None,
    train_mode: bool = False,
):
    list_disc_models = list()
    for params in list_disc_params:
        disc_path = f"{path}/{params['model_name']}/{params['model_id']}.pt"
        disc = get_model(
            model_name,
            model_params,
            device=device,
            path=disc_path,
            train_mode=train_mode,
        )
        list_disc_models.append(disc)
    return list_disc_models
