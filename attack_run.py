import copy
import warnings
warnings.filterwarnings('ignore')

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from src.data import load_data, transform_data, MyDataset
from src.attacks import attack_procedure
from src.utils import save_experiment, load_disc_model
from src.config import get_attack, load_disc_config


@hydra.main(config_path='config', config_name='attack_run_config', version_base=None)
def main(cfg: DictConfig):
    # load data
    X_train, y_train, X_test, y_test = load_data(cfg['dataset'])
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
    
    if cfg['use_disc_check']:
        disc_model_check = instantiate(cfg.disc_model_check).to(device)
        disc_model_check = load_disc_model(
            copy.deepcopy(disc_model_check),
            model_id=cfg['disc_check_params']['model_id'], 
            path=cfg['disc_path'], 
            model_name=cfg['disc_check_params']['model_name'], 
            device=device,
            )
        disc_model_check.train(cfg['train_mode'])
    else: 
        disc_model_check = None

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
                cfg['list_reg_model_params'],
                train_mode=cfg['train_mode']
                )  
        
        model_path = cfg['model_folder'] + f'model_{cfg["model_id_attack"]}_{cfg["dataset"]}.pth'
        model.load_state_dict(copy.deepcopy(torch.load(model_path)))

        aa_res_df, rej_curves_dict = attack_procedure(
            model = model, 
            loader = test_loader, 
            criterion = criterion,
            attack_func = attack_func,
            attack_params = attack_params,
            all_eps = cfg['all_eps'],
            n_steps = cfg['n_iterations'],
            n_objects = n_objects,
            train_mode = cfg['train_mode'],
            disc_model = disc_model_check,
            use_sigmoid = cfg['use_extra_sigmoid']
        )

        if not cfg['test_run']:
            print('Saving')
            save_experiment( aa_res_df, rej_curves_dict, path=cfg['save_path'], attack=cfg["attack_type"], dataset=cfg["dataset"], model_id=cfg["model_id_attack"], alpha=alpha)
        

if __name__=='__main__':
    main()



