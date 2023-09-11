import os
import pickle

import torch

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score)

from tqdm.auto import tqdm

def req_grad(model, state: bool = True) -> None:
    """Set requires_grad of all model parameters to the desired value.

    :param model: the model
    :param state: desired value for requires_grad
    """
    for param in model.parameters():
        param.requires_grad_(state)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, scheduler=None,
                 logger=None, n_epochs=30, print_every=5, device='cpu', multiclass=False):
        
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.criterion = criterion
    
        self.n_epoch = n_epochs
        self.optimizer = optimizer
        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = None

        self.device = device
        self.multiclass = multiclass
        self.print_every = print_every

        self.logger = logger
        self.dict_logging={}
        
        
    def _logging(self, data, epoch, mode='train',):
        
        for metric in self.dict_logging[mode].keys():
            self.dict_logging[mode][metric].append(data[metric])

            self.logger.add_scalar(metric+'/'+mode, data[metric], epoch)
    
    def train_model(self, early_stop_patience=None):

        if early_stop_patience and early_stop_patience != 'None':
            earl_stopper = EarlyStopper(early_stop_patience)

        metric_names = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'balance']
        self.dict_logging = {'train': {metric:[] for metric in metric_names},
                       'test': {metric:[] for metric in metric_names}}

        fill_line = 'Epoch {} train loss: {}; acc_train {}; test loss: {}; acc_test {}; f1_test {}; balance {}'

        for epoch in range(self.n_epoch):
            train_metrics_epoch = self._train_step()
            train_metrics_epoch = {met_name:met_val for met_name, met_val
                                   in zip(metric_names, train_metrics_epoch)}
            
            self._logging(train_metrics_epoch, epoch, mode='train')
            
            test_metrics_epoch = self._valid_step() 
            test_metrics_epoch = {met_name:met_val for met_name, met_val 
                       in zip(metric_names, test_metrics_epoch)}
            self._logging(test_metrics_epoch, epoch, mode='test')

            if epoch % self.print_every == 0:
                print_line = fill_line.format(epoch+1,
                                  round(train_metrics_epoch['loss'], 3),
                                  round(train_metrics_epoch['accuracy'], 3),
                                  round(test_metrics_epoch['loss'], 3),
                                  round(test_metrics_epoch['accuracy'], 3),
                                  round(test_metrics_epoch['f1'], 3),
                                  round(test_metrics_epoch['balance'], 3),
                                  )
                print(print_line)

            if early_stop_patience and early_stop_patience != 'None':
                res_early_stop = earl_stopper.early_stop(test_metrics_epoch['loss'])
                if res_early_stop:
                    break
        

    def _train_step(self):
        #req_grad(self.model)
        losses, n_batches = 0, 0
        
        y_all_pred = torch.tensor([])
        y_all_true = torch.tensor([])
        
        self.model.train(True)
        for x, labels in self.train_loader:

            self.optimizer.zero_grad()
            x = x.to(self.device)
            labels = labels.reshape(-1, 1).to(self.device)

            y_out = self.model(x)
            
            loss = self.criterion(y_out, labels) 

            loss.backward()     
            self.optimizer.step()
            losses += loss
            n_batches += 1
            
            if self.multiclass:
                y_pred = torch.argmax(y_out, axis=1)
            else:
                y_pred = torch.round(y_out)

            y_all_true = torch.cat((y_all_true, labels.cpu().detach()), dim=0)
            y_all_pred = torch.cat((y_all_pred, y_pred.cpu().detach()), dim=0)

        mean_loss = float((losses / n_batches).cpu().detach().numpy())

        if self.scheduler:
            self.scheduler.step()
            
        y_all_pred = y_all_pred.numpy().reshape([-1, 1])
        y_all_true = y_all_true.numpy().reshape([-1, 1])

        acc, pr, rec, f1 = self.calculate_metrics(y_all_true, y_all_pred)
        balance = np.sum(y_all_pred) / len(y_all_pred)
        return mean_loss, acc, pr, rec, f1, balance


    def _valid_step(self):
        
        y_all_pred = torch.tensor([])
        y_all_true = torch.tensor([])
        
        losses, n_batches = 0, 0
        self.model.eval()    
        for x, labels in self.test_loader:
            with torch.no_grad():
                x = x.to(self.device)
                labels = labels.reshape(-1, 1).to(self.device)


                y_out = self.model(x)
                #print(y_out)
                loss = self.criterion(y_out, labels)
                losses += loss
                n_batches += 1
                
                if self.multiclass:
                    y_pred = torch.argmax(y_out, axis=1)
                else:
                    y_pred = torch.round(y_out)

            y_all_true = torch.cat((y_all_true, labels.cpu().detach()), dim=0)
            y_all_pred = torch.cat((y_all_pred, y_pred.cpu().detach()), dim=0)

        mean_loss = float((losses / n_batches).cpu().detach().numpy())
        
        y_all_pred = y_all_pred.numpy().reshape([-1, 1])
        y_all_true = y_all_true.numpy().reshape([-1, 1])

        acc, pr, rec, f1 = self.calculate_metrics(y_all_true, y_all_pred)
        balance = np.sum(y_all_pred) / len(y_all_pred)
        return mean_loss, acc, pr, rec, f1, balance


    def calculate_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        pr = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return acc, pr, rec, f1
    
    def save_result(self, save_path, model_name):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        full_path = save_path + '/' + model_name
        torch.save(self.model.state_dict(), full_path + '.pth')

        with open(full_path+'_metrics.pickle', 'wb') as f:
            pickle.dump(self.dict_logging, f) 



        
################----------------------------------------------------------------------------------------------------------

def train_step(model, loader, criterion, device, optimizer, scheduler=None):
    losses, n_batches = 0, 0
    model.train(True)
    for x, labels in loader:

        optimizer.zero_grad()
        x = x.to(device)
        labels = labels.reshape(-1, 1).to(device)
        
        y_out = model(x)
        loss = criterion(y_out, labels) 
        
        loss.backward()     
        optimizer.step()
        losses += loss
        n_batches += 1

    mean_loss = losses / n_batches

    if scheduler:
        scheduler.step()
    
    return mean_loss

def valid_step(model, loader, criterion, device):
    
    losses, n_batches = 0, 0
    model.eval()    
    for x, labels in loader:
        with torch.no_grad():
            x = x.to(device)
            labels = labels.reshape(-1, 1).to(device)

            y_out = model(x)
            loss = criterion(y_out, labels)
            losses += loss

            n_batches += 1

    mean_loss = losses / n_batches
    return mean_loss


def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, pr, rec, f1
    
def estimate_epoch(loader, model=None, device='cpu', round_=True, multiclass=False):
    
    y_all_pred = torch.tensor([])
    y_all_true = torch.tensor([])
    
    for X, y_true in loader:
        X = X.to(device)
        y_pred = model(X)
        
        if multiclass:
            y_pred = torch.argmax(y_pred, axis=1)
        else:
            y_pred = torch.round(y_pred)
        
        y_all_true = torch.cat((y_all_true, y_true.cpu().detach()), dim=0)
        y_all_pred = torch.cat((y_all_pred, y_pred.cpu().detach()), dim=0)
        
    y_all_pred = y_all_pred.numpy().reshape([-1, 1])
    y_all_true = y_all_true.numpy().reshape([-1, 1])
    
    acc, pr, rec, f1 = calculate_metrics(y_all_true, y_all_pred)
    
    return acc, pr, rec, f1

def train_procedure(model, train_loader, test_loader, criterion, optimizer, scheduler=None,
                   num_epochs=30, step_print=5, device='cpu'):
    
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_step(model, train_loader, criterion, device, optimizer, scheduler)
        test_loss = valid_step(model, test_loader, criterion, device) 

        acc_train, _, _, f1_train = estimate_epoch(train_loader, model, device=device)
        acc_test, _, _, f1_test = estimate_epoch(test_loader, model, device=device)

        if epoch % step_print == 0:
            print(f'[Epoch {epoch + 1}] train loss: {train_loss:.3f}; acc_train {acc_train:.3f}; f1_train {f1_train:.3f}; test loss: {test_loss:.3f}; acc_test {acc_test:.3f}; f1_test {f1_test:.3f};')
            
    return model

