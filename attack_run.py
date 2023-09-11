import copy
import warnings
warnings.filterwarnings('ignore')

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from utils.data import load_Ford_A, transform_data, MyDataset
from models.models import LSTM_net

from utils.attacks import ifgsm_procedure
from utils.utils import save_experiment, load_disc_model
from utils.discrim_training import HideAttackExp
from utils.config import get_attack, load_disc_config
from utils.attacks import (fgsm_disc_attack, fgsm_attack, fgsm_reg_attack, 
simba_binary, simba_binary_reg, simba_binary_disc_reg)
from utils.TS2Vec.datautils import load_UCR



@hydra.main(config_path='config', config_name='attack_run_config', version_base=None)
def main(cfg: DictConfig):
    # load data
    if cfg['dataset'] == 'Ford_A':
        X_train, X_test, y_train, y_test = load_Ford_A()
    else:
        X_train, y_train, X_test, y_test = load_UCR(cfg['dataset'])
    
    X_train, X_test, y_train, y_test = transform_data(X_train, X_test, y_train, y_test, slice_data=cfg['slice'])
  
    test_loader = DataLoader(
        MyDataset(X_test, y_test), 
        batch_size=cfg['batch_size'] , 
        shuffle=False
        )

    criterion = torch.nn.BCELoss()
    n_objects = y_test.shape[0]
    device= torch.device(cfg['cuda'] if torch.cuda.is_available() else 'cpu')
    attack_func = get_attack(cfg['attack_type'])


    model = instantiate(cfg.attack_model).to(device)
    disc_model = instantiate(cfg.disc_model).to(device)
    
    disc_model_check = instantiate(cfg.disc_model_check).to(device)
    disc_model_check = load_disc_model(
        copy.deepcopy(disc_model_check),
        model_id=cfg['disc_check_params']['model_id'], 
        path=cfg['disc_path'], 
        model_name=cfg['disc_check_params']['model_name'], 
        device=device
        )
    disc_model_check.eval()

    alphas = [0]
    if 'reg' in cfg['attack_type'] or 'disc' in cfg['attack_type']:
        alphas = cfg['alphas']

    for alpha in tqdm(alphas):

        attack_params = dict()

        if 'reg' in cfg['attack_type'] :
            attack_params['alpha'] = alpha

        elif 'disc' in cfg['attack_type']:
            attack_params['alpha'] = alpha
            attack_params['disc_models'] = load_disc_config(
                copy.deepcopy(disc_model),
                cfg['disc_path'], 
                device, 
                cfg['list_reg_model_params']
                )  
            
            attack_params['disc_models'] = [model.eval() for model in  attack_params['disc_models']]
        
        model_path = cfg['model_folder'] + f'model_{cfg["model_id_attack"]}_{cfg["dataset"]}.pth'
        model.load_state_dict(copy.deepcopy(torch.load(model_path)))

        aa_res_df, rej_curves_dict = ifgsm_procedure(model=model, loader=test_loader, criterion=criterion,
                                                    attack_func=attack_func, attack_params=attack_params,
                                                    all_eps=cfg['all_eps'], n_steps=cfg['n_iterations'],
                                                    n_objects=n_objects, train_mode=cfg['train_mode'],
                                                    disc_model=disc_model_check)

        save_experiment( aa_res_df, rej_curves_dict, path=cfg['save_path'], attack=cfg["attack_type"], dataset=cfg["dataset"], model_id=cfg["model_id_attack"], alpha=alpha)
        

if __name__=='__main__':
    main()



