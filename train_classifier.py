import warnings

import time
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from clearml import Task

from src.data import MyDataset, load_data, transform_data
from src.training.train import Trainer
from src.utils import fix_seed, save_config, save_compiled_config

warnings.filterwarnings("ignore")

CONFIG_NAME = "train_classifier_config"
CONFIG_PATH = "config"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    start_time = time.time() 

    if not cfg["test_run"]:
        exp_name = cfg['exp_name'][1:] if cfg['exp_name'][0] == '_' else cfg['exp_name']
        model_save_name = f'model_{cfg["model"]["name"]}_{cfg["model_id"]}_{cfg["dataset"]["name"]}'

        if cfg['log_clearml']:

            if cfg['author'] == '':
                raise ValueError("You need to set your name in config")

            task = Task.init(
                project_name=cfg['clearml_project'],
                task_name=model_save_name,
                tags=[cfg["model"]["name"], cfg["dataset"]["name"], exp_name, cfg['author']]
            )
            task.upload_artifact(artifact_object=f'{cfg["save_path"]}/{model_save_name}.pt', name='model_weights.pt')
        else:
            task = None
        logger = SummaryWriter(cfg["save_path"] + "/tensorboard")
        save_config(cfg["save_path"], CONFIG_PATH, CONFIG_NAME, CONFIG_NAME)
        save_compiled_config(cfg, cfg["save_path"], model_save_name)
    else:
        logger = None

    print("trainig model", cfg['model_id'])

    augmentator = (
        [instantiate(trans) for trans in cfg["transform_data"]]
        if cfg["transform_data"]
        else None
    )

    # load data
    X_train, y_train, X_test, y_test = load_data(cfg["dataset"]['name'])

    if cfg["dataset"]["num_classes"] > 2:
        print(f"--- You have {cfg['dataset']['num_classes']} classes ---")
    
    X_train, X_test, y_train, y_test = transform_data(
        X_train,
        X_test,
        y_train,
        y_test,
        slice_data=cfg["slice"],
    )

    train_loader = DataLoader(
        MyDataset(X_train, y_train, augmentator),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    test_loader = DataLoader(
        MyDataset(X_test, y_test),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    const_params = {
        "logger": logger,
        "print_every": cfg["print_every"],
        "device": device,
        "seed": cfg['model_id'],
        "train_self_supervised": cfg['train_self_supervised'],
        "n_classes": cfg["dataset"]["num_classes"]
    }
    if cfg["enable_optimization"]:
        const_params['logger'] = None
        trainer = Trainer.initialize_with_optimization(
            train_loader, test_loader, cfg["optuna_optimizer"], const_params
        )
    else:
        trainer_params = dict(cfg["training_params"])
        trainer_params.update(const_params)
        trainer = Trainer.initialize_with_params(**trainer_params)

    trainer.train_model(train_loader, test_loader)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total wall clock time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    if not cfg["test_run"]:
        logger.close()
        trainer.save_result(cfg["save_path"], model_save_name, task)


if __name__ == "__main__":
    main()
