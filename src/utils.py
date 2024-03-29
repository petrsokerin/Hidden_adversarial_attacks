import copy
import os
import pickle
from typing import Dict
import shutil

import pandas as pd
import numpy as np
import torch
import random


def save_experiment(
        aa_res_df: pd.DataFrame,
        rej_curves_dict: Dict,
        config_name: str,
        path: str,
        attack: str,
        dataset: str,
        model_id: int,
        alpha: float
) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)

    if 'disc' in attack or 'reg' in attack:
        save_config(path, config_name, f"config_{dataset}_{model_id}_alpha={alpha}.yaml")
        aa_res_df.to_csv(path + f'/aa_res_{dataset}_{model_id}_alpha={alpha}.csv')
        with open(path + f'/rej_curves_dict_{dataset}_model_{model_id}_alpha={alpha}.pickle', 'wb') as file:
            pickle.dump(rej_curves_dict, file)

    else:
        save_config(path, config_name, f"config_{dataset}_{model_id}.yaml'")
        aa_res_df.to_csv(path + f'/aa_res_{dataset}_{model_id}.csv')
        with open(path + f'/rej_curves_dict_{dataset}_model_{model_id}.pickle', 'wb') as file:
            pickle.dump(rej_curves_dict, file)


def build_dataframe_metrics(experiment):
    df = pd.DataFrame()
    metrics_dict = experiment.dict_logging
    for split in metrics_dict.keys():
        df_loc = pd.DataFrame([])

        for key in metrics_dict[split].keys():
            df_loc[key] = metrics_dict[split][key]

        df_loc['split'] = split
        df_loc['iter'] = np.arange(1, len(df_loc) + 1)

        df_loc = df_loc[['iter', 'split'] + list(metrics_dict[split].keys())]

    df = pd.concat([df, df_loc])
    return df


def save_config(path, config_name, config_save_name) -> None:
    shutil.copytree(f'config/my_configs', path + '/config_folder', dirs_exist_ok = True)
    shutil.copyfile(f'config/my_configs/{config_name}.yaml', path + '/' + config_save_name)
    

def save_train_disc(experiment, config_name, model_id, cfg, save_csv=True):
    if 'prefix' not in cfg:
        cfg['prefix'] = ''

    if "reg" in cfg['attack_type'] or 'disc' in cfg['attack_type']:
        exp_name = f"{cfg['attack_type']}{cfg['prefix']}_eps={cfg['eps']}_alpha={cfg['alpha']}_nsteps={cfg['n_iterations']}"
    else:
        exp_name = f"{cfg['attack_type']}{cfg['prefix']}_eps={cfg['eps']}_nsteps={cfg['n_iterations']}"

    full_path = cfg['save_path'] + '/' + exp_name

    if not os.path.isdir(full_path):
        os.makedirs(full_path)

    if save_csv:
        df_res = build_dataframe_metrics(experiment)
        df_res.to_csv(full_path + '/' + f"{model_id}_logs.csv", index=None)

    model_weights_name = full_path + '/' + f"{model_id}.pt"
    torch.save(experiment.disc_model.state_dict(), model_weights_name)

    experiment.save_metrics_as_csv(full_path+'/' + f"{model_id}_logs.csv")
    save_config(full_path, config_name, f"{model_id}_config.yaml")


def save_train_classifier(model, save_path, model_name):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    full_path = save_path + '/' + model_name
    torch.save(model.state_dict(), full_path)


def load_disc_model(
    disc_model,
    path='results/FordA/Regular/Discriminator_pickle', 
    model_name='fgsm_attack_eps=0.03_nsteps=10',
    device='cpu', 
    model_id=0,
):
    path = fr'{path}/{model_name}/{model_id}.pt'

    disc_model = copy.deepcopy(disc_model)
    disc_model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    disc_model.to(device)
    disc_model.train(True)

    return disc_model


def fix_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

