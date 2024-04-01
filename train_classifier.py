import warnings

warnings.filterwarnings('ignore')

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data import load_data, transform_data, MyDataset
from src.training.train import Trainer
from src.utils import fix_seed, save_config

CONFIG_NAME = 'train_classifier_config'


@hydra.main(config_path='config/my_configs', config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    augmentator = [instantiate(trans) for trans in cfg['transform_data']] if cfg['transform_data'] else None

    # load data
    X_train, y_train, X_test, y_test = load_data(cfg['dataset'])
    if len(set(y_test)) > 2:
        return None
    X_train, X_test, y_train, y_test = transform_data(
        X_train,
        X_test,
        y_train,
        y_test,
        slice_data=cfg['slice'],
    )

    train_loader = DataLoader(
        MyDataset(X_train, y_train, augmentator),
        batch_size=cfg['batch_size'],
        shuffle=True
    )

    test_loader = DataLoader(
        MyDataset(X_test, y_test),
        batch_size=cfg['batch_size'],
        shuffle=False,
    )

    print('Size:', len(X_train), len(X_test))

    device = torch.device(cfg['cuda'] if torch.cuda.is_available() else 'cpu')

    for model_id in range(cfg['model_id_start'], cfg['model_id_finish']):
        print('trainig model', model_id)
        fix_seed(model_id)

        logger = SummaryWriter(cfg['save_path'] + '/tensorboard')
        
        const_params = {'logger': logger, 'print_every': cfg['print_every'], 'device': device, 'seed': model_id}
        if cfg['enable_optimization']:
            trainer = Trainer.initialize_with_optimization(
                train_loader, test_loader, cfg['optuna_optimizer'], const_params
            )

        else:
            trainer_params = dict(cfg['training_params'])
            trainer_params.update(const_params)
            trainer = Trainer.initialize_with_params(**trainer_params)

        trainer.train_model(train_loader, test_loader)
        logger.close()

        if not cfg['test_run']:
            print(cfg['save_path'])
            model_save_name = f'model_{model_id}_{cfg["dataset"]}'
            trainer.save_result(cfg['save_path'], model_save_name)
            save_config(cfg['save_path'], CONFIG_NAME, CONFIG_NAME)

            
if __name__=='__main__':
    main()