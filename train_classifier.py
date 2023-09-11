import copy
import warnings
warnings.filterwarnings('ignore')

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.data import load_Ford_A, transform_data, MyDataset
from utils.TS2Vec.datautils import load_UCR
from models.train import Trainer

from utils.utils import fix_seed


@hydra.main(config_path='config', config_name='train_classifier_config', version_base=None)
def main(cfg: DictConfig):

    # load data
    if cfg['dataset'] == 'Ford_A':
        X_train, X_test, y_train, y_test = load_Ford_A()
    else:
        X_train, y_train, X_test, y_test = load_UCR(cfg['dataset'])
    
    X_train, X_test, y_train, y_test = transform_data(X_train, X_test, y_train, y_test, slice_data=cfg['slice'])

    print(X_train.shape)

    train_loader = DataLoader(
        MyDataset(X_train, y_train), 
        batch_size=cfg['batch_size'] , 
        shuffle=True,
        drop_last=True
        )

    test_loader = DataLoader(
        MyDataset(X_test, y_test), 
        batch_size=cfg['batch_size'] , 
        shuffle=False
        )
    
    print(len(train_loader))

    criterion = torch.nn.BCELoss()
    device= torch.device(cfg['cuda'] if torch.cuda.is_available() else 'cpu')

    for model_id in range(cfg['model_id_start'], cfg['model_id_finish']):
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

        trainer.save_result(cfg['save_path'], model_name)
        
if __name__=='__main__':
    main()