import os
import shutil
from typing import Dict, Iterable, List

import torch
from src import attacks, estimation, models
from src.attacks import attack_scheduler

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
        raise ValueError(f"Scheduler with name {scheduler_name} is not implemented")


def get_attack_scheduler(
    attack_scheduler_name: str,
    attack: attacks.BaseIterativeAttack,
    attack_scheduler_params: Dict = None,
) -> attack_scheduler.AttackScheduler:
    if attack_scheduler_params is None:
        attack_scheduler_params = dict()
    try:
        return getattr(attack_scheduler, attack_scheduler_name)(
            attack, **attack_scheduler_params
        )
    except AttributeError:
        raise ValueError(
            f"Attack Scheduler with name {attack_scheduler_name} is not implemented"
        )


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
    path: str = "",
    train_mode: bool = False,
):
    list_disc_models = list()
    for model_conf in list_disc_params:
        if isinstance(model_conf, dict):
            weight_name = model_conf['model_id']
            model_folder_name = model_conf['model_name']
        else:
            weight_name = model_conf
            model_folder_name = model_conf
        disc_path = f"{path}/{model_folder_name}/{weight_name}.pt"
        disc = get_model(
            model_name,
            model_params,
            device=device,
            path=disc_path,
            train_mode=train_mode,
        )
        list_disc_models.append(disc)
    return list_disc_models

