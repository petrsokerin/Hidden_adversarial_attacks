import copy
import os
import pickle
import warnings
warnings.simplefilter("ignore")

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm
from src.training.discrim_training import HideAttackExp
from src.data import load_data, transform_data, build_dataloaders, MyDataset, Augmentator
from src.utils import save_train_disc
from src.config import get_attack, load_disc_config

CONFIG_NAME = 'train_disc_config'

@hydra.main(config_path='config/my_configs', config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):

    if cfg['test_run']:
        print('ATTENTION!!!! Results will not be saved. Set param test_run=False')

    augmentator = Augmentator([instantiate(trans) for trans in cfg['transform_data']]) 

    X_train, y_train, X_test, y_test = load_data(cfg['dataset'])
    X_train, X_test, y_train, y_test = transform_data(X_train, X_test, y_train, y_test, slice_data=cfg['slice'])

    train_loader, test_loader = build_dataloaders(X_train, X_test, y_train, y_test)

    device = torch.device(cfg['cuda'] if torch.cuda.is_available() else 'cpu')

    model_name = instantiate(cfg.attack_model).__class__.__name__
    cfg['model_folder'] = cfg['model_folder'] + f'{model_name}/'
    cfg['disc_path'] = cfg['disc_path'].format(model_name)
    cfg['save_path'] = cfg['save_path'].format(model_name) 

    print(cfg['alpha'], type(cfg['alpha']))

    for model_id in tqdm(cfg['model_ids']):

        attack_model = instantiate(cfg.attack_model).to(device)
        model_path = cfg['model_folder'] + f'model_{model_id}_{cfg["dataset"]}.pth'
        attack_model.load_state_dict(copy.deepcopy(torch.load(model_path)))
        
        attack_params = {'eps': cfg['eps']}

        attack_func = get_attack(cfg['attack_type'])

        discriminator_model = instantiate(cfg['disc_model']).to(device)

        if 'reg' in cfg['attack_type'] :
            attack_params['alpha'] = cfg['alpha']

        elif 'disc' in cfg['attack_type']:
            attack_params['alpha'] = cfg['alpha']
            attack_params['disc_models'] = load_disc_config(
                discriminator_model,
                cfg['disc_path'], 
                device, 
                cfg['list_reg_model_params']
            )  
            attack_params['use_sigmoid'] = cfg['use_extra_sigmoid']

        attack_train_params = {
            'attack_func': attack_func, 
            'attack_params': attack_params, 
            'criterion': torch.nn.BCELoss(), 
            'n_steps': cfg['n_iterations'],
            'train_mode': cfg['train_mode'],
        }
        attack_test_params = attack_train_params

        
        optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=cfg['lr'])
        disc_train_params = {
            'n_epoch': cfg['n_epochs'],
            'optimizer': optimizer,
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, cfg['step_lr'], gamma=cfg['gamma'])
        }      

        logger = SummaryWriter(cfg['save_path']+f'/tensorboard/{model_id}')
        experiment = HideAttackExp(
            attack_model,
            train_loader,
            test_loader,
            augmentator,
            attack_train_params,
            attack_test_params,
            discriminator_model,
            disc_train_params,
            logger = logger,
        )
        experiment.run(cfg['TS2Vec'], cfg['early_stop_patience'], cfg['verbose_ts2vec'])

        if not cfg['test_run']:
            save_train_disc(experiment, CONFIG_NAME, model_id, cfg)
            print('Success')


if __name__=='__main__':
    main()

