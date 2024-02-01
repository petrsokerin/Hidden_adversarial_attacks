import warnings
warnings.filterwarnings('ignore')

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data import load_data, transform_data, MyDataset
from src.training.train import Trainer
from src.utils import fix_seed

CONFIG_NAME = 'train_classifier_config'

@hydra.main(config_path='config/my_configs', config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):

    transforms = [instantiate(trans) for trans in cfg['transform_data']]  if cfg['transform_data'] else None   

    # load data
    X_train, y_train, X_test, y_test = load_data(cfg['dataset'])
    X_train, X_test, y_train, y_test = transform_data(
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        slice_data = cfg['slice'],
    )

    train_loader = DataLoader(
        MyDataset(X_train, y_train, transforms), 
        batch_size=cfg['batch_size'] , 
        shuffle=True
        )

    test_loader = DataLoader(
        MyDataset(X_test, y_test), 
        batch_size=cfg['batch_size'] , 
        shuffle=False,
        )
    
    print('N batches: ', len(train_loader))

    criterion = torch.nn.BCELoss()
    device = torch.device(cfg['cuda'] if torch.cuda.is_available() else 'cpu')
    print(device)

    for model_id in range(cfg['model_id_start'], cfg['model_id_finish']):
        print('trainig model', model_id)
        fix_seed(model_id)

        model_name = f'model_{model_id}_{cfg["dataset"]}'

        model = instantiate(cfg.model).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
        
        logger = SummaryWriter(cfg['save_path']+'/tensorboard')

        trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, scheduler, logger,
                    n_epochs=cfg['n_epochs'], print_every=cfg['print_every'], device=device)
        
        trainer.train_model()

        logger.close()

        if not cfg['test_run']:
            trainer.save_result(cfg['save_path'], model_name)
        
if __name__=='__main__':
    main()