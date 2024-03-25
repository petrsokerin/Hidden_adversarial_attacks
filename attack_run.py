import copy
import warnings
warnings.filterwarnings('ignore')

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tqdm.auto import tqdm

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import load_data, transform_data, MyDataset
from src.estimation.estimators import AttackEstimator
from src.attacks.attacks_procedures import BatchIterativeAttack
from src.utils import save_experiment
from src.config import *

CONFIG_NAME = 'attack_run_config'

@hydra.main(config_path='config/my_configs', config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):

    if cfg['test_run']:
        print('ATTENTION!!!! Results will not be saved. Set param test_run=False')
    
    # load data  
    print("Dataset", cfg['dataset'])
    X_train, y_train, X_test, y_test = load_data(cfg['dataset'])
    X_train, X_test, y_train, y_test = transform_data(X_train, X_test, y_train, y_test, slice_data=cfg['slice'])
  
    test_loader = DataLoader(
        MyDataset(X_test, y_test), 
        batch_size=cfg['batch_size'] , 
        shuffle=False
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

    if cfg['enable_optimization']:
        pass

    else:
        alphas = [0]
        if 'alpha' in cfg['attack']['attacks_params']:
            alphas = cfg['attack']['attacks_params']['alpha']

        for alpha in alphas: # tqdm(alphas):
            attack_metrics = pd.DataFrame()
            for eps in cfg['attack']['attacks_params']['eps']:  #tqdm(cfg['attack']['attacks_params']['eps']):

                attack_params = dict(cfg['attack']['attacks_params'])
                attack_params['model'] = attack_model
                attack_params['criterion'] = criterion
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

                attack = BatchIterativeAttack.initialize_with_params(cfg['attack']['name'], attack_params, estimator)
                attack.forward(test_loader) 
                results = attack.get_metrics()
                results['eps'] = eps
                attack_metrics = pd.concat([attack_metrics, results])

            if not cfg['test_run']:
                print('Saving')
                save_experiment(
                    attack_metrics,
                    config_name = CONFIG_NAME,
                    path = cfg['save_path'],
                    is_regularized = attack.is_regularized,
                    dataset = cfg["dataset"],
                    model_id = cfg["model_id_attack"],
                    alpha = alpha,
                )

if __name__=='__main__':
    main()



