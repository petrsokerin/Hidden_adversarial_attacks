import os
import warnings

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from clearml import Task

from src.config import get_attack, get_criterion, get_disc_list, get_model
from src.data import MyDataset, load_data, transform_data
from src.estimation.estimators import AttackEstimator
from src.utils import fix_seed, save_attack_metrics, save_config, save_compiled_config,weights_from_clearml_by_name

warnings.filterwarnings("ignore")

CONFIG_NAME = "attack_run_config"
CONFIG_PATH = "config"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    if cfg["test_run"]:
        print("ATTENTION!!!! Results will not be saved. Set param test_run=False")
        logger = None
    else:
        if cfg['author'] == '':
            raise ValueError("You need to set your name in config")

        attack_start_name = 'model_{}_{}_{}_attack_{}'.format(
            cfg["attack_model"]["name"],
            cfg["model_id_attack"],
            cfg["dataset"]["name"],
            cfg["attack"]["short_name"],
        )

        attack_add_name = ''
        for param in cfg['attack']['named_params']:
            attack_add_name += '__{}={}'.format(
                param,
                round(cfg['attack']['attack_params'][param], 4)
            )

        save_config(cfg["save_path"], CONFIG_PATH, CONFIG_NAME, attack_start_name)
        save_compiled_config(cfg, cfg["save_path"], attack_start_name + attack_add_name)


    # load data

    fix_seed(cfg['model_id_attack'])
    print("Dataset", cfg["dataset"]["name"])
    X_train, y_train, X_test, y_test = load_data(cfg["dataset"]["name"])
    X_train, X_test, y_train, y_test = transform_data(
        X_train, X_test, y_train, y_test, slice_data=cfg["slice"]
    )

    test_loader = DataLoader(
        MyDataset(X_test, y_test), batch_size=cfg["batch_size"], shuffle=False
    )

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    if cfg['pretrained']:
        project_name = cfg['project_weights']
        task_name = f"model_{cfg['attack_model']['name']}_{cfg['model_id_attack']}_{cfg['dataset']['name']}"
        path = weights_from_clearml_by_name(project_name=project_name, task_name=task_name)
        attack_model_path = os.path.join(path)
    else:
        attack_model_path = os.path.join(
        cfg["model_folder"],
        f"model_{cfg['attack_model']['name']}_{cfg['model_id_attack']}_{cfg['dataset']['name']}.pt")

    attack_model = get_model(
        cfg["attack_model"]["name"],
        cfg["attack_model"]["params"],
        path=attack_model_path,
        device=device,
        train_mode=cfg["attack_model"]["attack_train_mode"],
    )

    criterion = get_criterion(cfg["criterion_name"], cfg["criterion_params"])
    if cfg['pretrained_disc']:
        model_start_name = 'model_{}_{}_{}_attack_{}'.format(
            cfg["attack_model"]["name"],
            cfg["model_id_attack"],
            cfg["dataset"]["name"],
            'fgsm_attack',
        )
        model_add_name = cfg['disc_parametr']
        task_name = model_start_name + model_add_name
        path = weights_from_clearml_by_name(cfg['project_weights_disc'],task_name, 'disc_weights')
        
    else:
        path = cfg["disc_path"]

    if cfg["use_disc_check"]:
        disc_check_list = get_disc_list(
            model_name=cfg["disc_model_check"]["name"],
            model_params=cfg["disc_model_check"]["params"],
            list_disc_params=cfg["list_check_model_params"],
            device=device,
            path=cfg["disc_path"],
            train_mode=False,
            from_clearml=True
        )
    else:
        disc_check_list = None


    estimator = AttackEstimator(
        disc_check_list,
        cfg["metric_effect"],
        cfg["metric_hid"],
        batch_size=cfg["estimator_batch_size"],
    )

    attack_params = dict(cfg["attack"]["attack_params"])
    attack_params["model"] = attack_model
    attack_params["criterion"] = criterion
    attack_params["estimator"] = estimator

    if "list_reg_model_params" in cfg["attack"]:
        attack_params["disc_models"] = get_disc_list(
            model_name=cfg["disc_model_reg"]["name"],
            model_params=cfg["disc_model_reg"]["params"],
            list_disc_params=cfg["attack"]["list_reg_model_params"],
            device=device,
            path=path,
            train_mode=cfg["disc_model_reg"]["attack_train_mode"],
            from_clearml=True
        )

    attack = get_attack(cfg["attack"]["name"], attack_params)

    if cfg["enable_optimization"]:
        attack = attack.initialize_with_optimization(
            test_loader, cfg["optuna_optimizer"], attack_params
        )

        if not cfg["test_run"]:
            attack_add_name = ''
            for param in cfg['attack']['named_params']:
                attack_add_name += '__{}={}'.format(
                    param,
                    round(getattr(attack, param), 4)
                )

    if not cfg["test_run"]:
        attack_save_name = attack_start_name + attack_add_name
        exp_name = cfg['exp_name'][1:] if cfg['exp_name'][0] == '_' else cfg['exp_name']
        if cfg['log_clearml']:
            task = Task.init(
                project_name=cfg['clearml_project'],
                task_name=attack_save_name,
                tags=[
                    cfg["attack_model"]["name"],
                    cfg["dataset"]["name"],
                    cfg["attack"]["short_name"],
                    exp_name,
                    cfg['author'],
                ]
            )
        else:
            task = None

        logger = SummaryWriter(cfg["save_path"] + "/tensorboard")

    attack.apply_attack(test_loader, logger)

    if not cfg["test_run"]:
        print("Saving")
        attack_metrics = attack.get_metrics()
        for param in cfg['attack']['named_params']:
            attack_metrics[f'{param}_param'] = round(cfg['attack']['attack_params'][param], 4)
        save_attack_metrics(attack_metrics, cfg["save_path"], attack_save_name)
    os.remove(attack_model_path)
    os.remove(path)

if __name__ == "__main__":
    main()
