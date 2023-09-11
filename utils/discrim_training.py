from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader

from models.train import Trainer
from .attacks import IterGradAttack

class HideAttackExp:
    def __init__(self, attack_model, train_loader, test_loader, attack_train_params, 
                 attack_test_params, discriminator_model, disc_train_params, multiclass=False,
                 disc_print_every=1, logger=None):
        
        self.attack_loaders = {'train': train_loader,
                       'test': test_loader}
        self.attack_train = {'train':IterGradAttack(attack_model, train_loader, **attack_train_params),
                             'test': IterGradAttack(attack_model, test_loader, **attack_test_params)}
        
        self.disc_loaders = dict()
        
        self.attack_train_params = attack_train_params
        self.attack_test_params = attack_test_params
        
        
        self.eps = attack_train_params['attack_params']['eps']
        self.alpha = None
        if 'alpha' in attack_train_params['attack_params']:
            self.alpha = attack_train_params['attack_params']['alpha']
        self.n_steps = attack_train_params['n_steps']
        self.multiclass = multiclass

        
        self.attack_model = attack_model
        self.disc_model = discriminator_model
        

        self.disc_criterion = torch.nn.BCELoss()
        self.disc_n_epoch = disc_train_params['n_epoch']
        self.disc_batch_size = 64
        self.disc_optimizer = disc_train_params['optimizer']
        if 'scheduler' in disc_train_params.keys():
            self.disc_scheduler = disc_train_params['scheduler']
        else:
            self.disc_scheduler = None

        self.logger = logger
        self.disc_print_every = disc_print_every

        self.attack_device = next(attack_model.parameters()).device
        self.disc_device = next(discriminator_model.parameters()).device

        
    def run(self, TS2Vec=False, early_stop_patience=None, verbose_ts2vec=False):
        print("Generating adv data")
        self.get_disc_dataloaders(TS2Vec)
        print("Train discriminator")
        self.train_discriminator(TS2Vec, early_stop_patience, verbose_ts2vec=verbose_ts2vec)

        if TS2Vec:
            self.del_attack_model()
    
    def _generate_adv_data(self, mode='train', TS2Vec=False, batch_size=64):
        
        self.disc_batch_size = batch_size
        dataset_class = self.attack_train[mode].dataset_class

        X_adv, y_adv = self.attack_train[mode].run_iterations()
        X_orig = torch.tensor(self.attack_loaders[mode].dataset.X)
        X_adv = X_adv.squeeze(-1)

        adv_data_labels_shape = list(y_adv.shape)
        orig_data_labels_shape = adv_data_labels_shape
        orig_data_labels_shape[0] = len(X_orig)

        disc_labels_zeros = torch.zeros(orig_data_labels_shape) #True label class
        disc_labels_ones = torch.ones(adv_data_labels_shape) #True label class
        
        new_x = torch.concat([X_orig, X_adv], dim=0)
        new_y = torch.concat([disc_labels_zeros, disc_labels_ones], dim=0)

        suffle_status = mode == 'train'
        disc_loader = DataLoader(dataset_class(new_x, new_y), batch_size=batch_size, shuffle=suffle_status)
        self.disc_loaders[mode] = disc_loader
        self.X_train_disc = new_x.unsqueeze(-1).cpu().detach().numpy()
        return disc_loader
    
    def get_disc_dataloaders(self, TS2Vec):
        self._generate_adv_data('train', TS2Vec)
        self._generate_adv_data('test', TS2Vec)
        
    def _logging_train_disc(self, data, mode='train'):
        
        for metric in self.dict_logging[mode].keys():
            self.dict_logging[mode][metric].append(data[metric])

    def del_attack_model(self):
        del self.attack_model
        del self.attack_train
        del self.logger
    
    def train_discriminator(self, TS2Vec=False, early_stop_patience=None, verbose_ts2vec=False):

        if TS2Vec:
            print('Training TS2Vec')
            self.disc_model.train_embedding(self.X_train_disc, verbose=verbose_ts2vec)
            for param in self.disc_model.emd_model.parameters():
                param.requires_grad = False

        self.trainer = Trainer(
            self.disc_model, 
            self.disc_loaders['train'], 
            self.disc_loaders['test'], 
            self.disc_criterion, 
            self.disc_optimizer, 
            scheduler=self.disc_scheduler,
            logger=self.logger, 
            n_epochs=self.disc_n_epoch, 
            print_every=self.disc_print_every, 
            device=self.disc_device,
            multiclass=self.multiclass)
        
        self.trainer.train_model(early_stop_patience)

        self.dict_logging = self.trainer.dict_logging
        self.disc_model = self.trainer.model
        del self.trainer

    

