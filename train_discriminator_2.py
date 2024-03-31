import os
import warnings
warnings.filterwarnings('ignore')

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data import load_data, transform_data, MyDataset
from src.training.train import DiscTrainer

from src.estimation.estimators import AttackEstimator

from src.config import get_criterion, get_model, get_disc_list, get_attack
from src.utils import save_config

CONFIG_NAME = 'train_disc_config_2'


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

    device = torch.device(cfg['cuda'] if torch.cuda.is_available() else 'cpu')

    attack_model_path = os.path.join(
        cfg['model_folder'], 
        cfg['attack_model']['name'],
        f"model_{cfg['model_id_attack']}_{cfg['dataset']}.pth"
    )
    
    attack_model = get_model(
        cfg['attack_model']['name'], 
        cfg['attack_model']['params'], 
        path=attack_model_path, 
        device=device,
        train_mode=cfg['attack_model']['attack_train_mode']
    )

    criterion = get_criterion(cfg['criterion_name'], cfg['criterion_params'])
    
    disc_check_list = get_disc_list(
            model_name=cfg['disc_model_check']['name'], 
            model_params=cfg['disc_model_check']['params'],
            list_disc_params=cfg['list_check_model_params'], 
            device=device, 
            path=cfg['disc_path'], 
            train_mode=False
    ) if cfg['use_disc_check'] else None

    estimator = AttackEstimator(disc_check_list, cfg['metric_effect'])

    alphas = [0]
    if 'alpha' in cfg['attack']['attacks_params']:
            alphas = cfg['attack']['attacks_params']['alpha']

    for alpha in alphas:
        for eps in cfg['attack']['attacks_params']['eps']:
            print('----- Current epsilon:', eps, 
                  '\n----- Current alpha:', alpha)
            
            attack_params = dict(cfg['attack']['attacks_params'])
            attack_params['model'] = attack_model
            attack_params['criterion'] = criterion
            attack_params['estimator'] = estimator
            attack_params['alpha'] = alpha
            attack_params['eps'] = eps

            if 'list_reg_model_params' in cfg['attack']:
                attack_params['disc_models'] = get_disc_list(
                    model_name=cfg['disc_model_reg']['name'], 
                    model_params=cfg['disc_model_reg']['params'],
                    list_disc_params=cfg['attack']['list_reg_model_params'], 
                    device=device, 
                    path=cfg['disc_path'], 
                    train_mode=cfg['disc_model_reg']['attack_train_mode']
                )

            attack = get_attack(cfg['attack']['name'], attack_params)

            trainer_params = dict(cfg['training_params'])
            trainer_params['logger'] = SummaryWriter(cfg['save_path'] + '/tensorboard')
            trainer_params['attack'] = attack

            disc_trainer = DiscTrainer.initialize_with_params(**trainer_params)
            disc_trainer.train_model(train_loader, test_loader)

            if not cfg['test_run']:
                model_save_name = f'model_{cfg["model_id_attack"]}_{cfg["dataset"]}'
                new_save_path = cfg['save_path'] + '/' + f'eps={eps}_n_steps={cfg["attack"]["attacks_params"]["n_steps"]}'
                
                disc_trainer.save_result(new_save_path, model_save_name)
                save_config(new_save_path, CONFIG_NAME, CONFIG_NAME)
            
if __name__=='__main__':
    main()
