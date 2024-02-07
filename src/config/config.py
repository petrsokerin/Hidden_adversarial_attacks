from typing import List
import os
import shutil

from src.attacks import (fgsm_disc_attack, fgsm_attack, fgsm_reg_attack, 
simba_binary, simba_binary_reg, simba_binary_disc_reg, deepfool_attack, only_disc_attack, ascend_smax_disc_attack, fgsm_disc_smax_attack)
from src.utils import load_disc_model


def get_attack(attack_name):
    dict_attack = {
        'fgsm_attack': fgsm_attack, 
        'fgsm_reg_attack': fgsm_reg_attack, 
        'fgsm_disc_attack': fgsm_disc_attack, 
        'simba_binary': simba_binary, 
        'simba_binary_reg': simba_binary_reg, 
        'simba_binary_disc_reg': simba_binary_disc_reg, 
        'deepfool_attack': deepfool_attack,
        'only_disc_attack': only_disc_attack,
        'ascend_smax_disc_attack': ascend_smax_disc_attack,
        'fgsm_disc_smax_attack': fgsm_disc_smax_attack,
        'fgsm_clip_attack': fgsm_clip_attack,
    }

    if attack_name in dict_attack:
        return dict_attack[attack_name]
    else:
        raise ValueError("attack name isn't correct")


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
