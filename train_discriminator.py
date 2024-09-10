import os
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from clearml import Task

from src.config import get_criterion, get_disc_list, get_model
from src.data import MyDataset, load_data, transform_data
from src.estimation.estimators import AttackEstimator
from src.training.train import DiscTrainer
from src.utils import save_config, save_compiled_config

warnings.filterwarnings("ignore")

CONFIG_NAME = "train_disc_config"

torch.cuda.empty_cache()
@hydra.main(config_path="config/my_configs", config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):

    augmentator = (
        [instantiate(trans) for trans in cfg["transform_data"]]
        if cfg["transform_data"]
        else None
    )

    X_train, y_train, X_test, y_test = load_data(cfg["dataset"]['name'])

    if len(set(y_test)) > 2:
        return None

    X_train, X_test, y_train, y_test = transform_data(
        X_train,
        X_test,
        y_train,
        y_test,
        slice_data=cfg["slice"],
    )

    train_loader = DataLoader(
        MyDataset(X_train, y_train),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    test_loader = DataLoader(
        MyDataset(X_test, y_test),
        batch_size=cfg["batch_size"],
        shuffle=False,
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

    estimator = AttackEstimator(
        disc_check_list,
        cfg["metric_effect"],
        cfg["metric_hid"],
        batch_size=cfg["estimator_batch_size"],
    )

    for model_id in cfg["model_ids"]:

        if cfg["enable_optimization"]:
            attack_const_params = dict(cfg["attack"]["attack_params"])
            attack_const_params["model"] = attack_model
            attack_const_params["criterion"] = criterion
            attack_const_params["estimator"] = estimator
            attack_const_params["logger"] = None

            if "list_reg_model_params" in cfg["attack"]:
                attack_const_params["disc_models"] = get_disc_list(
                    model_name=cfg["disc_model_reg"]["name"],
                    model_params=cfg["disc_model_reg"]["params"],
                    list_disc_params=cfg["attack"]["list_reg_model_params"],
                    device=device,
                    path=cfg["disc_path"],
                    train_mode=cfg["disc_model_reg"]["attack_train_mode"],
                )

            const_params = {
                "attack_params": attack_const_params,
                "logger": None,
                "print_every": cfg["print_every"],
                "device": device,
                "seed": model_id,
                "train_self_supervised": cfg["train_self_supervised"],
            }

            disc_trainer = DiscTrainer.initialize_with_optimization(
                train_loader, test_loader, cfg["optuna_optimizer"], const_params
            )

            if not cfg["test_run"]:
                model_save_name = 'model_{}_{}_{}_attack_{}_eps={}_nsteps={}'.format(
                    cfg["model"]["name"],
                    model_id,
                    cfg["dataset"]["name"],
                    cfg["attack"]["short_name"],
                    round(disc_trainer.attack.eps, 4),
                    cfg["attack"]["attack_params"]["n_steps"]
                )
                task = Task.init(
                    project_name="AA_train_discriminator",
                    task_name=model_save_name,
                    tags=[cfg["model"]["name"], cfg["dataset"]["name"], cfg["attack"]["short_name"]]
                )
                logger = SummaryWriter(cfg["save_path"] + "/tensorboard")
            else:
                logger = None

            disc_trainer.train_model(train_loader, test_loader, augmentator, logger)

            if not cfg["test_run"]:
                new_save_path = os.path.join(cfg["save_path"], model_save_name)
                save_config(new_save_path, CONFIG_NAME, CONFIG_NAME)
                disc_trainer.save_result(new_save_path, model_save_name, task)


        else:
            alphas = [0]

            if "alpha" in cfg["attack"]["attack_params"]:
                alphas = cfg["attack"]["attack_params"]["alpha"]

            for alpha in alphas:
                for eps in cfg["attack"]["attack_params"]["eps"]:
                    print(
                        "----- Current epsilon:", eps, "\n----- Current alpha:", alpha
                    )

                    if not cfg["test_run"]:
                        model_save_name = 'model_{}_{}_{}_attack_{}_eps={}_nsteps={}_alpha={}'.format(
                            cfg["model"]["name"],
                            model_id,
                            cfg["dataset"]["name"],
                            cfg["attack"]["short_name"],
                            eps,
                            cfg["attack"]["attack_params"]["n_steps"],
                            alpha,
                        )
                        task = Task.init(
                            project_name="AA_train_discriminator",
                            task_name=model_save_name,
                            tags=[cfg["model"]["name"], cfg["dataset"]["name"], cfg["attack"]["short_name"]]
                        )
                        logger = SummaryWriter(cfg["save_path"] + "/tensorboard")

                        new_save_path = os.path.join(cfg["save_path"], model_save_name)
                        save_config(new_save_path, CONFIG_NAME, CONFIG_NAME)
                        save_compiled_config(cfg, new_save_path)

                    else:
                        logger = None

                    attack_params = dict(cfg["attack"]["attack_params"])
                    attack_params["model"] = attack_model
                    attack_params["criterion"] = criterion
                    attack_params["estimator"] = estimator
                    attack_params["alpha"] = alpha
                    attack_params["eps"] = eps
                    attack_params["logger"] = None

                    if "list_reg_model_params" in cfg["attack"]:
                        attack_params["disc_models"] = get_disc_list(
                            model_name=cfg["disc_model_reg"]["name"],
                            model_params=cfg["disc_model_reg"]["params"],
                            list_disc_params=cfg["attack"]["list_reg_model_params"],
                            device=device,
                            path=cfg["disc_path"],
                            train_mode=cfg["disc_model_reg"]["attack_train_mode"],
                        )

                    trainer_params = dict(cfg["training_params"])
                    trainer_params["logger"] = logger
                    trainer_params["device"] = device
                    trainer_params["seed"] = model_id
                    trainer_params["train_self_supervised"] = cfg[
                        "train_self_supervised"
                    ]

                    trainer_params["attack_name"] = cfg["attack"]["name"]
                    trainer_params["attack_params"] = attack_params

                    disc_trainer = DiscTrainer.initialize_with_params(**trainer_params)
                    disc_trainer.train_model(train_loader, test_loader, augmentator)

                    if not cfg["test_run"]:
                        disc_trainer.save_result(new_save_path, model_save_name, task)



if __name__ == "__main__":
    main()
