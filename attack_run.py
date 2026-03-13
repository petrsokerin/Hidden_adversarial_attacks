import os
import warnings

import time
import hydra
import torch
import numpy as np
import yaml
from omegaconf import OmegaConf
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from clearml import Task

from src.config import get_attack, get_criterion, get_disc_list, get_model
from src.data import MyDataset, load_data, transform_data
from src.estimation.estimators import AttackEstimator
from src.utils import fix_seed, save_attack_metrics, save_config, save_compiled_config,weights_from_clearml_by_name
# from src.training.train_attacker import train_atk_model
from src.training.train import GenAttackTrainer


warnings.filterwarnings("ignore")

CONFIG_NAME = "attack_run_config"
CONFIG_PATH = "config"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    start_time = time.time()

    if cfg["test_run"]:
        print("ATTENTION!!!! Results will not be saved. Set param test_run=False")
        logger = None
    else:
        if cfg['log_clearml'] and cfg['author'] == '':
            raise ValueError("You need to set your name in config")

        # Определяем имя модели для названия атаки
        if cfg['attack'].get('is_trainable', False):
            # Для генеративных атак используем learning_target_model
            model_name_for_title = cfg["learning_target_model"]["name"]
            model_id_for_title = cfg["model_id_learning"]
        else:
            # Для обычных атак используем attack_model
            model_name_for_title = cfg["attack_model"]["name"]
            model_id_for_title = cfg["model_id_attack"]

        attack_base_name = 'model_{}_{}_{}_attack_{}'.format(
            model_name_for_title,
            model_id_for_title,
            cfg["dataset"]["name"],
            cfg["attack"]["short_name"],
        )

        attack_named_params_str = ''
        for param in cfg['attack']['named_params']:
            try:
                attack_named_params_str += '__{}={}'.format(
                    param,
                    round(cfg['attack']['attack_params'][param], 4)
                )
            except:
                attack_named_params_str += '__{}={}'.format(
                    param,
                    cfg['attack']['attack_params'][param]
                )


        save_config(cfg["save_path"], CONFIG_PATH, CONFIG_NAME, attack_base_name)
        attack_save_name = attack_base_name + attack_named_params_str
        save_compiled_config(cfg, cfg["save_path"], attack_save_name)


    # load data

    fix_seed(cfg['seed'])
    print("Dataset", cfg["dataset"]["name"])
    X_train, y_train, X_test, y_test = load_data(cfg["dataset"]["name"])
    X_train, X_test, y_train, y_test = transform_data(
        X_train, X_test, y_train, y_test, slice_data=cfg["slice"]
    )

    test_loader = DataLoader(
        MyDataset(X_test, y_test), batch_size=cfg["batch_size"], shuffle=False
    )

    device = torch.device(cfg["device"])

    # Загружаем attack_model (для обычных атак)
    if cfg['load_weights_classifier']:
        project_name = cfg['project_weights']
        task_name = f"model_{cfg['attack_model']['name']}_{cfg['model_id_attack']}_{cfg['dataset']['name']}"
        path = weights_from_clearml_by_name(project_name=project_name, task_name=task_name)
        attack_model_path = os.path.join(path)
    else:
        attack_model_path = os.path.join(
            cfg["model_folder"],
            f"model_{cfg['attack_model']['name']}_{cfg['model_id_attack']}_{cfg['dataset']['name']}.pt"
        )

    attack_model = get_model(
        cfg["attack_model"]["name"],
        cfg["attack_model"]["params"],
        path=attack_model_path,
        device=device,
        train_mode=cfg["attack_model"]["attack_train_mode"],
    )

    # learning, inference target models for gen atk
    learning_target_model = None
    inference_target_model = None

    if cfg['attack'].get('is_trainable', False):
        # load learning_target_model (for training gen attack)
        if cfg.get('load_weights_learning', False):
            project_name = cfg['project_weights']
            task_name = f"model_{cfg['learning_target_model']['name']}_{cfg['model_id_learning']}_{cfg['dataset']['name']}"
            path = weights_from_clearml_by_name(project_name=project_name, task_name=task_name)
            learning_model_path = os.path.join(path)
        else:
            learning_model_path = os.path.join(
                cfg["learning_model_folder"],
                f"model_{cfg['learning_target_model']['name']}_{cfg['model_id_learning']}_{cfg['dataset']['name']}.pt"
            )

        learning_target_model = get_model(
            cfg["learning_target_model"]["name"],
            cfg["learning_target_model"]["params"],
            path=learning_model_path,
            device=device,
            train_mode=cfg["learning_target_model"]["attack_train_mode"],
        )

        # load inference_target_model (for final evaluation)
        if cfg.get('load_weights_inference', False):
            project_name = cfg['project_weights']
            task_name = f"model_{cfg['inference_target_model']['name']}_{cfg['model_id_inference']}_{cfg['dataset']['name']}"
            path = weights_from_clearml_by_name(project_name=project_name, task_name=task_name)
            inference_model_path = os.path.join(path)
        else:
            inference_model_path = os.path.join(
                cfg["inference_model_folder"],
                f"model_{cfg['inference_target_model']['name']}_{cfg['model_id_inference']}_{cfg['dataset']['name']}.pt"
            )

        inference_target_model = get_model(
            cfg["inference_target_model"]["name"],
            cfg["inference_target_model"]["params"],
            path=inference_model_path,
            device=device,
            train_mode=False,  # always eval mode for inference
        )

    criterion = get_criterion(cfg["criterion_name"], cfg["criterion_params"])
    if cfg['load_weights_disc']:
        path = cfg['project_weights_disc']

    else:
        path = cfg["disc_path"]

    if cfg["use_disc_check"]:
        disc_check_list = get_disc_list(
            model_name=cfg["disc_model_check"]["name"],
            model_params=cfg["disc_model_check"]["params"],
            list_disc_params=cfg["list_check_model_params"],
            device=device,
            path=path,
            train_mode=False,
            from_clearml=cfg['load_weights_disc']
        )
    else:
        disc_check_list = None


    estimator = AttackEstimator(
        disc_check_list,
        cfg["metric_effect"],
        cfg["metric_hid"],
        batch_size=cfg["estimator_batch_size"],
        n_classes = cfg["dataset"]["num_classes"]
    )

    attack_params = dict(cfg["attack"]["attack_params"])
    attack_params["model"] = attack_model
    attack_params["criterion"] = criterion
    attack_params["estimator"] = estimator
    attack_params["n_classes"] = cfg["dataset"]["num_classes"]

    if "list_reg_model_params" in cfg["attack"]:
        attack_params["disc_models"] = get_disc_list(
            model_name=cfg["disc_model_reg"]["name"],
            model_params=cfg["disc_model_reg"]["params"],
            list_disc_params=cfg["attack"]["list_reg_model_params"],
            device=device,
            path=path,
            train_mode=cfg["disc_model_reg"]["attack_train_mode"],
            from_clearml=cfg['load_weights_disc']
        )


    if not cfg["test_run"]:
        exp_name = cfg['exp_name'][1:] if cfg['exp_name'][0] == '_' else cfg['exp_name']
        if cfg['log_clearml']:
            task = Task.init(
                project_name=cfg['clearml_project'],
                task_name=attack_save_name,
                tags=[
                    model_name_for_title,
                    cfg["dataset"]["name"],
                    cfg["attack"]["short_name"],
                    exp_name,
                    cfg['author'],
                ]
            )
        else:
            task = None

        logger = SummaryWriter(cfg["save_path"] + "/tensorboard")


    is_learnable = cfg['attack'].get('is_trainable', False)

    # Инициализируем переменные для генеративной модели атаки
    gen_attack_model_path = None
    gen_attack_model_from_clearml = False

    if not is_learnable:
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
    else:
        # load-name for gen attack model weights (like other models)
        gen_attack_base_name = f"gen_attack_{cfg['gen_attack_model']['name']}_{cfg['model_id_gen_attack']}_{cfg['learning_target_model']['name']}_{cfg['dataset']['name']}_{cfg['attack']['short_name']}"

        # Добавляем параметры атаки (attack_add_name уже сформирован выше)
        gen_attack_model_name = gen_attack_base_name + attack_named_params_str

        # Определяем источник загрузки (clearml или локально)
        # Путь всегда формируется, независимо от флага (как для других моделей)
        if cfg.get('load_weights_gen_attack', False) and cfg.get('project_weights_gen_attack'):
            # Загружаем из clearml
            project_name = cfg['project_weights_gen_attack']
            task_name = gen_attack_model_name
            try:
                path = weights_from_clearml_by_name(project_name=project_name, task_name=task_name)
                gen_attack_model_path = os.path.join(path)
                gen_attack_model_from_clearml = True
            except Exception as e:
                print(f"Warning: Could not load gen attack model weights from clearml: {e}. Trying local path.")
                gen_attack_model_path = os.path.join(
                    cfg["gen_attack_model_folder"],
                    f"{gen_attack_model_name}.pt"
                )
                gen_attack_model_from_clearml = False
        else:
            # Загружаем локально (по умолчанию, как и для других моделей)
            gen_attack_model_path = os.path.join(
                cfg["gen_attack_model_folder"],
                f"{gen_attack_model_name}.pt"
            )
            gen_attack_model_from_clearml = False

        # Проверяем существование файла и информируем пользователя
        print(f"\n=== Gen Attack Model Loading ===")
        print(f"Expected model name: {gen_attack_model_name}")
        print(f"Model folder: {cfg['gen_attack_model_folder']}")
        print(f"Full path: {gen_attack_model_path}")

        gen_model_loaded = False
        if gen_attack_model_path and os.path.exists(gen_attack_model_path):
            print(f"Found gen attack model weights, loading from: {gen_attack_model_path}")
            gen_model_loaded = True
        elif gen_attack_model_path:
            print(f"Warning: Gen attack model weights not found at {gen_attack_model_path}")
            # Проверяем, есть ли файлы в директории
            if os.path.isdir(cfg["gen_attack_model_folder"]):
                existing_files = os.listdir(cfg["gen_attack_model_folder"])
                print(f"  Existing files in directory: {existing_files}")
            print(f"  Will train from scratch.")
            gen_attack_model_path = None  # Не передаем путь, чтобы модель создалась с нуля
        else:
            print(f"Warning: No path specified, will train from scratch")
        print(f"===============================\n")


        # Для генеративных атак используем learning_target_model вместо attack_model
        attack_params['model'] = learning_target_model

        if gen_model_loaded:
            print(f"✓ Gen attack model loaded successfully. Skipping training.")
            # attack = get_attack(cfg["attack"]["name"], attack_params)
            trainer_params = dict(cfg["attack"]["training_params"])
            trainer_params.update({
                "attack_name": cfg["attack"]["name"],
                "attack_params": attack_params,
                "logger": None,
                "print_every": cfg["attack"]["training_params"]["print_every"],
                "device": device,
                "seed": cfg['seed'],
                "train_self_supervised": cfg["attack"]["training_params"]["train_self_supervised"],
                "gen_model_name": cfg["gen_attack_model"]["name"],
                "gen_model_params": cfg["gen_attack_model"]["params"],
                "gen_model_path": gen_attack_model_path,
            })
            attack_trainer = GenAttackTrainer.initialize_with_params(**trainer_params)
            attack = attack_trainer.attack
        else:
            # Обучаем модель только если она не была загружена
            training_train_loader = DataLoader(
                MyDataset(X_train, y_train), batch_size=cfg["attack"]["batch_size"], shuffle=True
            )

            training_test_loader = DataLoader(
                MyDataset(X_test, y_test), batch_size=cfg["attack"]["batch_size"], shuffle=False
            )

            trainer_logger = SummaryWriter(cfg["save_path"] + "/training_tensorboard")

            const_trainer_params = dict(cfg["attack"]["training_params"])  # <-- базовые training_params
            const_trainer_params.update({
                "attack_name":  cfg["attack"]["name"],
                "attack_params": attack_params,
                "logger": trainer_logger,
                "print_every": cfg["attack"]["training_params"]["print_every"],
                "device": device,
                "seed": cfg['seed'],
                "train_self_supervised": cfg["attack"]["training_params"]["train_self_supervised"],
                "gen_model_name": cfg["gen_attack_model"]["name"],
                "gen_model_params": cfg["gen_attack_model"]["params"],
                "gen_model_path": gen_attack_model_path if gen_model_loaded else None,
            })

            if cfg["enable_optimization"]:
                const_trainer_params['logger'] = None
                attack_trainer = GenAttackTrainer.initialize_with_optimization(
                    training_train_loader, training_test_loader, cfg["optuna_optimizer"], const_trainer_params
                )
            else:
                trainer_params = dict(cfg["attack"]["training_params"])
                trainer_params.update(const_trainer_params)
                attack_trainer = GenAttackTrainer.initialize_with_params(**trainer_params)

            # attack = attack_trainer.train_model(training_train_loader, training_test_loader)
            attack_trainer.train_model(training_train_loader, training_test_loader)
            attack = attack_trainer.attack

            # Сохраняем веса генеративной модели атаки и метрики после обучения
            if not cfg["test_run"]:
                gen_attack_base_name = f"gen_attack_{cfg['gen_attack_model']['name']}_{cfg['model_id_gen_attack']}_{cfg['learning_target_model']['name']}_{cfg['dataset']['name']}_{cfg['attack']['short_name']}"
                gen_attack_model_name = gen_attack_base_name + attack_named_params_str

                attack_trainer.save_result(
                    save_path=cfg["gen_attack_model_folder"],
                    model_name=gen_attack_model_name,
                    task=task if cfg['log_clearml'] else None
                )
                print(f"Gen attack model weights and metrics saved to: {cfg['gen_attack_model_folder']}/{gen_attack_model_name}")

                # OPTUNA CFG BLOCK START
                # Сохраняем конфиг атакующей модели
                attack_config_path = os.path.join(cfg["gen_attack_model_folder"], gen_attack_model_name + "_config.yaml")
                attack_cfg = OmegaConf.to_container(cfg["attack"], resolve=True)
                attack_cfg["gen_attack_model"] = OmegaConf.to_container(cfg["gen_attack_model"], resolve=True)
                attack_cfg["training_params"] = OmegaConf.to_container(cfg["attack"].get("training_params", {}), resolve=True)
                with open(attack_config_path, "w") as f:
                    yaml.dump(attack_cfg, f, default_flow_style=False, allow_unicode=True)
                print(f"Attack config saved to: {attack_config_path}")


                # Сохраняем лучшие параметры Optuna
                if hasattr(attack_trainer, 'optuna_best_params'):
                    attack_cfg["optuna_best_params"] = attack_trainer.optuna_best_params
                
                # Сохраняем реальные training_params из обученного трейнера
                attack_cfg["actual_training_params"] = {
                    "n_epochs": attack_trainer.n_epochs,
                    "alpha_l2": attack_trainer.alpha_l2,
                    "optimizer_name": type(attack_trainer.optimizer).__name__,
                    "optimizer_params": {
                        "lr": attack_trainer.optimizer.param_groups[0]["lr"],
                    },
                    "scheduler_name": type(attack_trainer.scheduler).__name__ if attack_trainer.scheduler else "None",
                }
                if attack_trainer.scheduler and hasattr(attack_trainer.scheduler, 'gamma'):
                    attack_cfg["actual_training_params"]["scheduler_params"] = {
                        "gamma": attack_trainer.scheduler.gamma,
                        "step_size": attack_trainer.scheduler.step_size,
                    }

                with open(attack_config_path, "w") as f:
                    yaml.dump(attack_cfg, f, default_flow_style=False, allow_unicode=True)
                print(f"Attack config saved to: {attack_config_path}")

                 # OPTUNA CFG BLOCK END

        # train_atk_model(attack.attacker, attack_model, train_loader, device=device)

    # Заменяем модель на inference перед финальной оценкой (для генеративных атак)
    if cfg['attack'].get('is_trainable', False) and inference_target_model is not None:
        print(f"Replacing model: {type(attack.model).__name__} -> {type(inference_target_model).__name__}")
        attack.set_inference_model(inference_target_model)
        print(f"Model replaced successfully!")

    # Применяем атаку и получаем атакованные данные
    X_adv = attack.apply_attack(test_loader, logger)

    # Выводим финальные метрики после атаки на inference модели
    if not cfg["test_run"] and inference_target_model is not None:
        print(f"\nFinal attack metrics on inference_target_model ({cfg['inference_target_model']['name']}):")

        # Создаем временный estimator для inference модели
        inference_estimator = AttackEstimator(
            None,  # без discriminator check
            cfg["metric_effect"],
            cfg["metric_hid"],
            batch_size=cfg["estimator_batch_size"],
            n_classes=cfg["dataset"]["num_classes"]
        )

        # Вычисляем метрики на inference модели
        with torch.no_grad():
            y_true = test_loader.dataset.y
            y_pred_orig = inference_target_model(test_loader.dataset.X.unsqueeze(-1).to(device))
            y_pred_adv = inference_target_model(X_adv.to(device))

            # Подготавливаем данные для estimator
            if cfg["dataset"]["num_classes"] > 2:
                y_pred_orig_classes = y_pred_orig.argmax(dim=-1).cpu()
                y_pred_adv_classes = y_pred_adv.argmax(dim=-1).cpu()
            else:
                y_pred_orig_classes = (y_pred_orig > 0.5).float().cpu()
                y_pred_adv_classes = (y_pred_adv > 0.5).float().cpu()

            # Подготавливаем данные для estimator (как в procedures.py)
            X_orig = test_loader.dataset.X
            if X_orig.dim() == 2:
                X_orig = X_orig.unsqueeze(-1)
            if X_adv.dim() == 2:
                X_adv = X_adv.unsqueeze(-1)

            # get metrics from attack on inference model 
            #(before inference_estimator.estimate() was used)           
            attack_metrics = attack.get_metrics()
            print(attack_metrics.to_string())

    elif not cfg["test_run"]:
        print(f"\nFinal attack metrics on learning_target_model ({cfg['learning_target_model']['name']}):")

        attack_metrics = attack.get_metrics()
        print(attack_metrics.to_string())

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total wall clock time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    if not cfg["test_run"]:
        print("Saving")
        attack_metrics = attack.get_metrics()
        for param in cfg['attack']['named_params']:
            attack_metrics[f'{param}_param'] = round(cfg['attack']['attack_params'][param], 4)
        save_attack_metrics(attack_metrics, cfg["save_path"], attack_save_name)
        if cfg['load_weights_classifier']:
            os.remove(attack_model_path)
        if cfg.get('load_weights_learning', False):
            os.remove(learning_model_path)
        if cfg.get('load_weights_inference', False) and inference_target_model is not None:
            os.remove(inference_model_path)
        if cfg.get('load_weights_gen_attack', False) and gen_attack_model_from_clearml and gen_attack_model_path:
            os.remove(gen_attack_model_path)
        if cfg['delete_weights_disc']:
            target_folder = 'loaded_clearml/disc_weights/'

            for file_name in os.listdir(target_folder):
                file_path = os.path.join(target_folder, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)


if __name__ == "__main__":
    main()
