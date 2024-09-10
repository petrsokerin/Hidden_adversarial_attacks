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
from src.utils import save_attack_metrics, save_config, save_compiled_config

warnings.filterwarnings("ignore")

CONFIG_NAME = "attack_run_config"

@hydra.main(config_path="config/my_configs", config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    if cfg["test_run"]:
        print("ATTENTION!!!! Results will not be saved. Set param test_run=False")
    else:
        save_config(
            cfg["save_path"],
            CONFIG_NAME,
            f"config_{cfg['dataset']['name']}_{cfg['model_id_attack']}",
        )
        save_compiled_config(cfg,cfg["save_path"])
    # load data
    print("Dataset", cfg["dataset"]["name"])
    X_train, y_train, X_test, y_test = load_data(cfg["dataset"]["name"])
    X_train, X_test, y_train, y_test = transform_data(
        X_train, X_test, y_train, y_test, slice_data=cfg["slice"]
    )

    test_loader = DataLoader(
        MyDataset(X_test, y_test), batch_size=cfg["batch_size"], shuffle=False
    )

    device = torch.device(cfg["cuda"] if torch.cuda.is_available() else "cpu")

    attack_model_path = os.path.join(
        cfg["model_folder"],
        f"model_{cfg['model_id_attack']}_{cfg['dataset']['name']}.pt",
    )

    attack_model = get_model(
        cfg["attack_model"]["name"],
        cfg["attack_model"]["params"],
        path=attack_model_path,
        device=device,
        train_mode=cfg["attack_model"]["attack_train_mode"],
    )

    criterion = get_criterion(cfg["criterion_name"], cfg["criterion_params"])

    if cfg["use_disc_check"]:
        disc_check_list = get_disc_list(
            model_name=cfg["disc_model_check"]["name"],
            model_params=cfg["disc_model_check"]["params"],
            list_disc_params=cfg["list_check_model_params"],
            device=device,
            path=cfg["disc_path"],
            train_mode=False,
        )
    else:
        disc_check_list = None

    if cfg["enable_optimization"]:
        estimator = AttackEstimator(
            disc_check_list,
            cfg["metric_effect"],
            cfg["metric_hid"],
            batch_size=cfg["estimator_batch_size"],
        )
        const_params = dict(cfg["attack"]["attack_params"])
        const_params["model"] = attack_model
        const_params["criterion"] = criterion
        const_params["estimator"] = estimator

        if "list_reg_model_params" in cfg["attack"]:
            const_params["disc_models"] = get_disc_list(
                model_name=cfg["disc_model_reg"]["name"],
                model_params=cfg["disc_model_reg"]["params"],
                list_disc_params=cfg["attack"]["list_reg_model_params"],
                device=device,
                path=cfg["disc_path"],
                train_mode=cfg["disc_model_reg"]["attack_train_mode"],
            )

        attack = get_attack(cfg["attack"]["name"], const_params)
        attack = attack.initialize_with_optimization(
            test_loader, cfg["optuna_optimizer"], const_params
        )

        if not cfg["test_run"]:
            attack_save_name = 'model_{}_{}_{}_attack_{}_eps={}_nsteps={}_alpha={}'.format(
                cfg["attack_model"]["name"],
                cfg["model_id_attack"],
                cfg["dataset"]["name"],
                cfg["attack"]["short_name"],
                eps,
                cfg["attack"]["attack_params"]["n_steps"],
                alpha,
            )
            task = Task.init(
                project_name="AA_attack_run",
                task_name=attack_save_name,
                tags=[cfg["attack_model"]["name"], cfg["dataset"]["name"], cfg["attack"]["short_name"]]
            )
            logger = SummaryWriter(cfg["save_path"] + "/tensorboard")
        attack.apply_attack(test_loader, logger)

        attack_metrics = attack.get_metrics()
        attack_metrics["eps"] = attack.eps

        alpha = attack.alpha if getattr(attack, "alpha", None) else 0

        if not cfg["test_run"]:
            print("Saving")
            save_attack_metrics(attack_metrics, cfg["save_path"], attack_save_name)

    else:
        alphas = [0]
        if "alpha" in cfg["attack"]["attack_params"]:
            alphas = cfg["attack"]["attack_params"]["alpha"]

        for alpha in alphas:
            attack_metrics = pd.DataFrame()
            for eps in cfg["attack"]["attack_params"]["eps"]:

                if not cfg["test_run"]:
                    attack_save_name = 'model_{}_{}_{}_attack_{}_eps={}_nsteps={}_alpha={}'.format(
                        cfg["attack_model"]["name"],
                        cfg["model_id_attack"],
                        cfg["dataset"]["name"],
                        cfg["attack"]["short_name"],
                        eps,
                        cfg["attack"]["attack_params"]["n_steps"],
                        alpha,
                    )
                    task = Task.init(
                        project_name="AA_attack_run",
                        task_name=attack_save_name,
                        tags=[cfg["attack_model"]["name"], cfg["dataset"]["name"], cfg["attack"]["short_name"]]
                    )
                    logger = SummaryWriter(cfg["save_path"] + "/tensorboard")

                else:
                    logger = None

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
                attack_params["alpha"] = alpha
                attack_params["eps"] = eps
                attack_params["logger"] = logger

                if "list_reg_model_params" in cfg["attack"]:
                    attack_params["disc_models"] = get_disc_list(
                        model_name=cfg["disc_model_reg"]["name"],
                        model_params=cfg["disc_model_reg"]["params"],
                        list_disc_params=cfg["attack"]["list_reg_model_params"],
                        device=device,
                        path=cfg["disc_path"],
                        train_mode=cfg["disc_model_reg"]["attack_train_mode"],
                    )

                attack = get_attack(cfg["attack"]["name"], attack_params)
                attack.apply_attack(test_loader)
                results = attack.get_metrics()
                results["eps"] = eps
                attack_metrics = pd.concat([attack_metrics, results])

            if not cfg["test_run"]:
                print("Saving")
                save_attack_metrics(attack_metrics, cfg["save_path"], attack_save_name)


if __name__ == "__main__":
    main()
