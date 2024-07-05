import copy
import os
import random
from datetime import datetime
import shutil
import yaml
from typing import Any, Dict, Mapping
from datetime import datetime
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from optuna.trial import Trial
from src.data import load_data
from src.estimation.utils import calculate_roughness


def save_config(path, config_name: str, config_save_name: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)

    shutil.copytree("config/my_configs", path + "/config_folder", dirs_exist_ok=True)
    shutil.copyfile(
        f"config/my_configs/{config_name}.yaml", path + "/" + config_save_name + '.yaml'
    )

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Создаем словарь с метаданными
    metadata = {"date": date, "time": time}

    # Создаем файл metadata.yaml в указанной директории
    metadata_path = os.path.join(path, "metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)





def req_grad(model, state: bool = True) -> None:
    """Set requires_grad of all model parameters to the desired value.

    :param model: the model
    :param state: desired value for requires_grad
    """
    for param in model.parameters():
        param.requires_grad_(state)


def save_attack_metrics(
    attack_metrics: pd.DataFrame,
    path: str,
    is_regularized: bool,
    dataset: str,
    model_id: int,
    alpha: float,
) -> None:

    if not os.path.isdir(path):
        os.makedirs(path)
    attack_metrics = attack_metrics.round(4)

    if is_regularized:
        attack_metrics.to_csv(path + f"/aa_res_{dataset}_{model_id}_alpha={alpha}.csv")

    else:
        attack_metrics.to_csv(path + f"/aa_res_{dataset}_{model_id}.csv")


def get_optuna_param_for_type(
    param_name: str,
    optuna_type: str,
    optuna_params: Dict,
    initial_model_parameters: Dict,
    trial: Trial,
):
    if optuna_type == "int":
        initial_model_parameters[param_name] = trial.suggest_int(**optuna_params)
    elif optuna_type == "float":
        initial_model_parameters[param_name] = trial.suggest_float(**optuna_params)
    elif optuna_type == "choice":
        initial_model_parameters[param_name] = trial.suggest_categorical(
            **optuna_params
        )
    elif optuna_type == "const":
        initial_model_parameters[param_name] = optuna_params["value"]

    return initial_model_parameters, trial


def update_one_param(
    new_param_name: str, new_param_value: Any, final_params: Dict
) -> Dict:
    for def_param_name, def_param_value in final_params.items():
        if isinstance(def_param_value, dict):
            final_params[def_param_name] = update_one_param(
                new_param_name, new_param_value, def_param_value
            )
        else:
            if new_param_name == def_param_name:
                final_params[new_param_name] = new_param_value
                # final_params.update({best_param_name: best_param_value})
    return final_params


def update_dict_params(original_params: Dict, new_params: Dict) -> Dict:
    final_params = copy.deepcopy(original_params)
    for new_param_name, new_param_value in new_params.items():
        final_best_params = update_one_param(
            new_param_name, new_param_value, final_params
        )
    return final_best_params


def update_params_with_attack_params(params: Dict, new_params: Dict) -> Dict:
    if "attack_params" in params:
        for param in new_params:
            if param == "attack_params":
                params["attack_params"].update(new_params["attack_params"])
            else:
                params[param] = new_params[param]
    else:
        params.update(new_params)
    return params


def collect_default_params(params_vary: DictConfig) -> Dict:
    initial_model_parameters = {}

    for param_name, param_value in params_vary.items():
        if "optuna_type" in param_value:
            if param_value["optuna_type"] == "const":
                initial_model_parameters[param_name] = param_value["value"]
            else:
                initial_model_parameters[param_name] = None
        else:
            sub_init = collect_default_params(param_value)
            initial_model_parameters[param_name] = sub_init

    return initial_model_parameters


def get_optimization_dict(params_vary: DictConfig, trial: Trial) -> Mapping[str, Any]:
    initial_model_parameters = {}

    for param_name, param_value in params_vary.items():
        if "optuna_type" in param_value:
            optuna_params = dict(param_value)
            del optuna_params["optuna_type"]
            optuna_params["name"] = param_name

            initial_model_parameters, trial = get_optuna_param_for_type(
                param_name=param_name,
                optuna_type=param_value["optuna_type"],
                optuna_params=optuna_params,
                initial_model_parameters=initial_model_parameters,
                trial=trial,
            )
        else:
            sub_init, trial = get_optimization_dict(param_value, trial)
            initial_model_parameters[param_name] = sub_init

    return initial_model_parameters, trial


def build_dataframe_metrics(experiment):
    df = pd.DataFrame()
    metrics_dict = experiment.dict_logging
    for split in metrics_dict.keys():
        df_loc = pd.DataFrame([])

        for key in metrics_dict[split].keys():
            df_loc[key] = metrics_dict[split][key]

        df_loc["split"] = split
        df_loc["iter"] = np.arange(1, len(df_loc) + 1)

        df_loc = df_loc[["iter", "split"] + list(metrics_dict[split].keys())]

    df = pd.concat([df, df_loc])
    return df


def save_train_disc(experiment, config_name, model_id, cfg, save_csv=True):
    if "prefix" not in cfg:
        cfg["prefix"] = ""

    if "reg" in cfg["attack_type"] or "disc" in cfg["attack_type"]:
        exp_name = f"{cfg['attack_type']}{cfg['prefix']}_eps={cfg['eps']}_alpha={cfg['alpha']}_nsteps={cfg['n_iterations']}"
    else:
        exp_name = f"{cfg['attack_type']}{cfg['prefix']}_eps={cfg['eps']}_nsteps={cfg['n_iterations']}"

    full_path = cfg["save_path"] + "/" + exp_name

    if not os.path.isdir(full_path):
        os.makedirs(full_path)

    if save_csv:
        df_res = build_dataframe_metrics(experiment)
        df_res.to_csv(full_path + "/" + f"{model_id}_logs.csv", index=None)

    model_weights_name = full_path + "/" + f"{model_id}.pt"
    torch.save(experiment.disc_model.state_dict(), model_weights_name)

    experiment.save_metrics_as_csv(full_path + "/" + f"{model_id}_logs.csv")
    save_config(full_path, config_name, f"{model_id}_config.yaml")


def save_train_classifier(model, save_path, model_name):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    full_path = save_path + "/" + model_name
    torch.save(model.state_dict(), full_path)


def fix_seed(seed: int) -> None:
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior for cudnn
    torch.backends.cudnn.deterministic = True


def calc_stats(data):
    stats = {}
    stats['object_count'] = data.shape[0]
    stats['max'] = float(data.max())
    stats['mean'] = float(data.mean())
    stats['min'] = float(data.min())
    stats['roughness'] = calculate_roughness(data)

    return stats


def get_dataset_stats(dataset_name, path='config/my_configs/dataset/'):
    X_train, y_train, X_test, y_test = load_data(dataset_name)

    stats = {}
    stats['name'] = dataset_name
    stats['num_classes'] = len(np.unique(y_train))
    stats['seq_len'] = X_train.shape[1]
    stats['total_object_count'] = len(y_train)+len(y_test)

    train = calc_stats(X_train)
    test = calc_stats(X_test)

    stats['train'] = train
    stats['test'] = test

    with open(path + f'{dataset_name}.yaml', 'w+') as f:
        yaml.dump(stats, f, sort_keys=False)

def save_compiled_config(cfg, path):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # save_path = os.path.join(os.getcwd(), "loggs")
    # save_path = os.path.join(path, "loggs")
    os.makedirs(path, exist_ok=True)
    config_filename = f"loggs_{timestamp}.yaml"
    config_path = os.path.join(path, config_filename)

    # Convert OmegaConf config to dictionary and add timestamp
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict['save_timestamp'] = timestamp

    # Save the updated configuration to YAML file
    with open(config_path, "w") as file:
        yaml.dump(cfg_dict, file)

    print(f"Compiled configuration saved to: {config_path}")
