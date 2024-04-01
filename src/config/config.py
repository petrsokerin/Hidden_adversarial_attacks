import os
import shutil
from ast import Dict

import torch

from src import attacks, estimation, models


def get_estimator(estimator_name: str, estimator_params: Dict):
    if estimator_params is None:
        estimator_params = dict()
    try:
        return getattr(estimation, estimator_name)(**estimator_params)
    except AttributeError:
        raise ValueError(f"Estimator with name {estimator_name} is not implemented")


def get_attack(attack_name: str, attack_params: Dict):
    if attack_params is None:
        attack_params = dict()
    try:
        return getattr(attacks, attack_name)(**attack_params)
    except AttributeError:
        raise ValueError(f"Attack with name {attack_name} is not implemented")


def get_model(model_name, model_params, device="cpu", path=None, train_mode=False):
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


def get_criterion(criterion_name, criterion_params=None):
    if criterion_params is None:
        criterion_params = dict()
    try:
        return getattr(torch.nn, criterion_name)(**criterion_params)
    except AttributeError:
        raise ValueError(f"Criterion with name {criterion_name} is not implemented")


def get_optimizer(optimizer_name, model_params, optimizer_params=None):
    if optimizer_params is None:
        optimizer_params = dict()
    try:
        return getattr(torch.optim, optimizer_name)(model_params, **optimizer_params)
    except AttributeError:
        raise ValueError(f"Optimizer with name {optimizer_name} is not implemented")


def get_scheduler(scheduler_name, optimizer, scheduler_params=None):
    if scheduler_params is None:
        scheduler_params = dict()
    try:
        return getattr(torch.optim.lr_scheduler, scheduler_name)(
            optimizer, **scheduler_params
        )
    except AttributeError:
        raise ValueError(f"Optimizer with name {scheduler_name} is not implemented")


def save_config(path, config_load_name, config_save_name):
    if not os.path.isdir(path):
        os.makedirs(path)

    shutil.copyfile(
        f"config/{config_load_name}.yaml", path + f"/{config_save_name}.yaml"
    )


def get_disc_list(
    model_name,
    model_params,
    list_disc_params,
    device="cpu",
    path=None,
    train_mode=False,
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


# def load_disc_model(
#     disc_model: torch.nn.Module,
#     path: str,
#     model_name: str,
#     device: str = "cpu",
#     model_id: int = 0,
# ):
#     path = rf"{path}/{model_name}/{model_id}.pt"

#     disc_model = copy.deepcopy(disc_model)
#     disc_model.load_state_dict(torch.load(path, map_location=torch.device(device)))
#     disc_model.to(device)
#     disc_model.train(True)

#     return disc_model


# def load_disc_config(
#     disc_model: torch.nn.Module,
#     path: str,
#     device: str,
#     list_disc_params: List[Dict],
#     train_mode: bool = True,
# ) -> List[torch.nn.Module]:
#     list_disc_models = list()

#     for params in list_disc_params:
#         model = load_disc_model(disc_model, device=device, path=path, **params)
#         model.train(train_mode)
#         list_disc_models.append(model)

#     return list_disc_models
