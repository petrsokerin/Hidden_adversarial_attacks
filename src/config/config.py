from typing import List
import os
import shutil

import torch

from src.attacks import (fgsm_disc_attack, fgsm_attack, fgsm_reg_attack,
simba_binary, simba_binary_reg, simba_binary_disc_reg, deepfool_attack, fgsm_clip_attack, only_disc_attack, ascend_smax_disc_attack, fgsm_disc_smax_attack)
from src.utils import load_disc_model
from src import models, attacks, estimation

# def get_attack(attack_name, attack_params):
#     dict_attack = {
#         'fgsm_attack': fgsm_attack, 
#         'fgsm_reg_attack': fgsm_reg_attack, 
#         'fgsm_disc_attack': fgsm_disc_attack, 
#         'simba_binary': simba_binary, 
#         'simba_binary_reg': simba_binary_reg, 
#         'simba_binary_disc_reg': simba_binary_disc_reg, 
#         'deepfool_attack': deepfool_attack,
#         'only_disc_attack': only_disc_attack,
#         'ascend_smax_disc_attack': ascend_smax_disc_attack,
#         'fgsm_disc_smax_attack': fgsm_disc_smax_attack,
#     }

#     if attack_name in dict_attack:
#         return dict_attack[attack_name]
#     else:
#         raise ValueError(f"Model with name {attack_name} is not implemented")

def get_estimator(estimator_name, estimator_params):
    if estimator_params is None:
        estimator_params = dict()
    try: 
        getattr(estimation, estimator_name)(**estimator_params)
    except AttributeError:
        raise ValueError(f"Model with name {estimator_name} is not implemented")



def get_attack(attack_name, attack_params):
    if model_params is None:
        model_params = dict()
    try: 
        getattr(attacks, attack_name)(**attack_params)
    except AttributeError:
        raise ValueError(f"Model with name {attack_name} is not implemented")

def get_model(model_name, model_params):
    if model_params is None:
        model_params = dict()
    try: 
        getattr(models, model_name)(**model_params)
    except AttributeError:
        raise ValueError(f"Model with name {model_name} is not implemented")

    
def get_criterion(criterion_name, criterion_params=None):
    if criterion_params is None:
        criterion_params = dict()
    try: 
        getattr(torch.nn, criterion_name)(**criterion_params)
    except AttributeError:
        raise ValueError(f"Criterion with name {criterion_name} is not implemented")
    
    
def get_optimizer(optimizer_name, model_params, optimizer_params=None):
    if optimizer_params is None:
        optimizer_params = dict()
    try: 
        getattr(torch.optim, optimizer_name)(model_params, **optimizer_params)
    except AttributeError:
        raise ValueError(f"Optimizer with name {optimizer_name} is not implemented")

    
def get_scheduler(scheduler_name, optimizer, scheduler_params=None):
    if scheduler_params is None:
        scheduler_params = dict()
    try: 
        getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_params)
    except AttributeError:
        raise ValueError(f"Optimizer with name {scheduler_name} is not implemented")
    

# def get_attack(attack_name):
#     dict_attack = {
#         'fgsm_attack': fgsm_attack, 
#         'fgsm_reg_attack': fgsm_reg_attack, 
#         'fgsm_disc_attack': fgsm_disc_attack, 
#         'simba_binary': simba_binary, 
#         'simba_binary_reg': simba_binary_reg, 
#         'simba_binary_disc_reg': simba_binary_disc_reg, 
#         'deepfool_attack': deepfool_attack,
#         'only_disc_attack': only_disc_attack,
#         'ascend_smax_disc_attack': ascend_smax_disc_attack,
#         'fgsm_disc_smax_attack': fgsm_disc_smax_attack,
#     }

#     if attack_name in dict_attack:
#         return dict_attack[attack_name]
#     else:
#         raise ValueError("attack name isn't correct")


def save_config(path, config_load_name, config_save_name):
    if not os.path.isdir(path):
        os.makedirs(path)

    shutil.copyfile(f'config/{config_load_name}.yaml', path + f'/{config_save_name}.yaml')


def load_disc_config(
        disc_model,
        path: str,
        device: str,
        list_disc_params: List,
        train_mode: bool = True,
) -> List:
    list_disc_models = list()

    for params in list_disc_params:
        model = load_disc_model(disc_model, device=device, path=path, **params)
        model.train(train_mode)
        list_disc_models.append(model)

    return list_disc_models
