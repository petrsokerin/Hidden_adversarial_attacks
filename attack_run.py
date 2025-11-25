import os
import warnings

import time
import hydra
import torch
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
from src.visualization.attack_visualizer import save_attack_visualizations

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

        attack_start_name = 'model_{}_{}_{}_attack_{}'.format(
            model_name_for_title,
            model_id_for_title,
            cfg["dataset"]["name"],
            cfg["attack"]["short_name"],
        )

        attack_add_name = ''
        for param in cfg['attack']['named_params']:
            try:
                attack_add_name += '__{}={}'.format(
                    param,
                    round(cfg['attack']['attack_params'][param], 4)
                )
            except:
                attack_add_name += '__{}={}'.format(
                    param,
                    cfg['attack']['attack_params'][param]
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
        attack_save_name = attack_start_name + attack_add_name
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
        gen_attack_model_name = gen_attack_base_name + attack_add_name
        
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
        
        gen_model = get_model(
            cfg["gen_attack_model"]["name"],
            cfg["gen_attack_model"]["params"],
            device=device,
            path=gen_attack_model_path,
        )

        # Для генеративных атак используем learning_target_model вместо attack_model
        attack_params['model'] = learning_target_model
        attack_params['gen_model'] = gen_model
        
        if gen_model_loaded:
            print(f"✓ Gen attack model loaded successfully. Skipping training.")
            attack = get_attack(cfg["attack"]["name"], attack_params)
        else:
            # Обучаем модель только если она не была загружена
            training_train_loader = DataLoader(
                MyDataset(X_train, y_train), batch_size=cfg["attack"]["batch_size"], shuffle=True
            )

            training_test_loader = DataLoader(
                MyDataset(X_test, y_test), batch_size=cfg["attack"]["batch_size"], shuffle=False
            )

            trainer_logger = SummaryWriter(cfg["save_path"] + "/training_tensorboard")

            const_trainer_params = {
                "attack_name":  cfg["attack"]["name"],
                "attack_params": attack_params,
                "logger": trainer_logger,
                "print_every": cfg["attack"]["training_params"]["print_every"],
                "device": device,
                "seed": cfg['model_id_attack'],
                "train_self_supervised": cfg["attack"]["training_params"]["train_self_supervised"],
            }
            if cfg["enable_optimization"]:
                const_trainer_params['logger'] = None
                attack_trainer = GenAttackTrainer.initialize_with_optimization(
                    training_train_loader, training_test_loader, cfg["optuna_optimizer"], const_trainer_params
                )
            else:
                trainer_params = dict(cfg["attack"]["training_params"])
                trainer_params.update(const_trainer_params)
                attack_trainer = GenAttackTrainer.initialize_with_params(**trainer_params)

            attack = attack_trainer.train_model(training_train_loader, training_test_loader)
            
            # Сохраняем веса генеративной модели атаки и метрики после обучения
            if not cfg["test_run"]:
                gen_attack_base_name = f"gen_attack_{cfg['gen_attack_model']['name']}_{cfg['model_id_gen_attack']}_{cfg['learning_target_model']['name']}_{cfg['dataset']['name']}_{cfg['attack']['short_name']}"
                gen_attack_model_name = gen_attack_base_name + attack_add_name
                
                attack_trainer.save_result(
                    save_path=cfg["gen_attack_model_folder"],
                    model_name=gen_attack_model_name,
                    task=task if cfg['log_clearml'] else None
                )
                print(f"Gen attack model weights and metrics saved to: {cfg['gen_attack_model_folder']}/{gen_attack_model_name}")

        # train_atk_model(attack.attacker, attack_model, train_loader, device=device)

    # Заменяем модель на inference перед финальной оценкой (для генеративных атак)
    if cfg['attack'].get('is_trainable', False) and inference_target_model is not None:
        print(f"Replacing model: {type(attack.model).__name__} -> {type(inference_target_model).__name__}")
        attack.set_inference_model(inference_target_model)
        print(f"Model replaced successfully!")

    # Применяем атаку и получаем атакованные данные
    X_adv = attack.apply_attack(test_loader, logger)

    # Визуализация результатов атаки
    if not cfg["test_run"]:
        vis_model = inference_target_model or attack_model
        if vis_model is None:
            print("Visualization skipped: no model available.")
        else:
            was_training = vis_model.training
            vis_model.eval()

            X_orig_vis = test_loader.dataset.X
            if X_orig_vis.dim() == 2:
                X_orig_vis = X_orig_vis.unsqueeze(-1)

            X_adv_vis = X_adv
            if X_adv_vis.dim() == 2:
                X_adv_vis = X_adv_vis.unsqueeze(-1)

            with torch.no_grad():
                y_pred_orig_vis = vis_model(X_orig_vis.to(device))
                y_pred_adv_vis = vis_model(X_adv_vis.to(device))

            if was_training:
                vis_model.train()

            def _scalar_preds(preds: torch.Tensor) -> torch.Tensor:
                if preds.dim() == 1:
                    return preds
                if preds.shape[-1] == 1:
                    return preds.squeeze(-1)
                return torch.softmax(preds, dim=-1).max(dim=-1).values

            y_pred_orig_vis = _scalar_preds(y_pred_orig_vis)
            y_pred_adv_vis = _scalar_preds(y_pred_adv_vis)

            vis_save_dir = os.path.join(
                cfg["save_path"], "visualizations", attack_save_name
            )
            max_samples = cfg.visualization_max_samples if "visualization_max_samples" in cfg else 5
            save_attack_visualizations(
                vis_save_dir,
                X_orig_vis.cpu(),
                X_adv_vis.cpu(),
                test_loader.dataset.y.cpu(),
                y_pred_orig_vis.cpu(),
                y_pred_adv_vis.cpu(),
                max_samples=max_samples,
            )

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
            
            # Вычисляем метрики через estimator (как в procedures.py)
            metrics_line = inference_estimator.estimate(
                y_true.numpy(), 
                y_pred_adv.cpu().numpy(), 
                y_pred_adv_classes.numpy(), 
                y_pred_orig_classes.numpy(), 
                X_orig.numpy(), 
                X_adv.numpy(), 
                0
            )
            
            # Выводим метрики
            for metric_name, metric_value in zip(inference_estimator.metrics_names, metrics_line):
                print(f"  {metric_name}: {metric_value:.4f}")
    elif not cfg["test_run"]:
        print(f"\nFinal attack metrics on learning_target_model ({cfg['learning_target_model']['name']}):")
        attack_metrics = attack.get_metrics()
        for metric_name, metric_value in attack_metrics.items():
            if isinstance(metric_value, (int, float)):
                print(f"  {metric_name}: {metric_value:.4f}")

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
