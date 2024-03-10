import os
import pickle
from functools import partial


from tqdm.auto import tqdm
import torch
import optuna
from optuna.trial import Trial
from hydra.utils import instantiate
from omegaconf import DictConfig

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

from src.utils import get_optimization_lists


def req_grad(model, state: bool = True) -> None:
    """Set requires_grad of all model parameters to the desired value.

    :param model: the model
    :param state: desired value for requires_grad
    """
    for param in model.parameters():
        param.requires_grad_(state)


class EarlyStopper:
    def __init__(
            self,
            patience: int = 1,
            min_delta: float = 0.0,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def get_criterion(criterion_name, criterion_params={}):
    if criterion_name == 'BCE':
        return torch.nn.BCELoss(**criterion_params)
    elif criterion_name == 'CrossEntropy':
        return torch.nn.CrossEntropyLossLoss(**criterion_params)
    else:
        raise ValueError("Only BCE and CrossEntropy losses are implemented")
    
def get_optimizer(optimizer_name, model_params, optimizer_params={}):
    if optimizer_name == 'Adam':
        return torch.optim.Adam(model_params, **optimizer_params)
    elif optimizer_name == 'SGD':
        return torch.optim.SGD(model_params, **optimizer_params)
    else:
        raise ValueError("Only Adam and SGD optimizers are implemented")
    
def get_scheduler(scheduler_name, optimizer, scheduler_params):
    if scheduler_name == 'None':
        return None
    elif scheduler_name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    else:
        raise ValueError("Only None and StepLR optimizers are implemented")


class Trainer:
    def __init__(
            self, 
            model, 
            train_loader, 
            test_loader, 
            criterion_name='BCE', 
            criterioin_params={},
            optimizer_name='Adam', 
            optimizer_params={'lr': 1e-3}, 
            scheduler_name='None',
            scheduler_params={},
            logger=None, 
            n_epochs=30, 
            print_every=5, 
            device='cpu', 
            multiclass=False):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = get_criterion(criterion_name, criterioin_params)
        self.optimizer = get_optimizer(optimizer_name, model.parameters, optimizer_params)
        self.scheduler = get_scheduler(scheduler_name, self.optimizer, scheduler_params)

        self.n_epoch = n_epochs

        self.device = device
        self.multiclass = multiclass
        self.print_every = print_every

        self.logger = logger
        self.dict_logging = {}

    @staticmethod
    def init_with_params(
        train_loader, 
        test_loader,
        model_params,
    ):
        
        pass

    @staticmethod
    def initialize_with_optimization(
            self,
            optuna_params,
            train_loader,
            valid_loader,
    ):

        study = optuna.create_study(
            direction="maximize",
            sampler=instantiate(optuna_params["sampler"]),
            pruner=instantiate(optuna_params["pruner"]),
        )
        study.optimize(
            partial(
                Trainer.objective,
                params_vary=optuna_params["hyperparameters_vary"],
                k_opt=optuna_params["k_optimization"],
                train_loader=train_loader,
                valid_loader=valid_loader,
            ),
            n_trials=optuna_params["n_trials"],
        )
        best_params = study.best_params.copy()
        if "const" in optuna_params["hyperparameters_vary"]:
            best_params.update(optuna_params["hyperparameters_vary"]["const"])

        self.logger.info("Best parameters are - %s", best_params)
        return Trainer.initialize_with_params(train_loader, valid_loader, best_params)

    @staticmethod
    def objective(
        trial: Trial,
        params_vary: DictConfig,
        train_loader,
        valid_loader,
    ) -> float:
        """
        Objective function for hyperparameter optimization using Optuna.

        Args:
            trial (Trial): Optuna trial object.
            params_vary (DictConfig): Hyperparameters to optimize.
            k_opt (int): Number of top-k items to evaluate.
            train_loader (TrainDataLoader): train dataloader from recbole package.
            valid_loader (FullSortEvalDataLoader): valid dataloader from recbole package.

        Returns:
            float: Mean rank of the model evaluated on the validation set.
        """

        initial_model_parameters = get_optimization_lists(params_vary, trial)
        model = Trainer.initialize_with_params(
            train_loader, valid_loader, initial_model_parameters
        )
        model.fit(train_loader, valid_loader)
        top_100_items = model.recommend_k(valid_loader, k=100)
        recbole_val_lists = np.array(
            [
                tensor.tolist()
                for tensor in valid_loader.uid2positive_item
                if tensor is not None
            ],
            dtype=object,
        )
        metrics = []
        for k in k_opt:
            metrics.append(
                normalized_discounted_cumulative_gain(
                    top_100_items, recbole_val_lists, k
                )
            )
        return np.mean(metrics)

    
    def _logging(self, data, epoch, mode='train', ):

        for metric in self.dict_logging[mode].keys():
            self.dict_logging[mode][metric].append(data[metric])

            self.logger.add_scalar(metric + '/' + mode, data[metric], epoch)

    def train_model(self, early_stop_patience=None):

        if early_stop_patience and early_stop_patience != 'None':
            earl_stopper = EarlyStopper(early_stop_patience)

        metric_names = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'balance']
        self.dict_logging = {'train': {metric: [] for metric in metric_names},
                             'test': {metric: [] for metric in metric_names}}

        fill_line = 'Epoch {} train loss: {}; acc_train {}; test loss: {}; acc_test {}; f1_test {}; balance {}'

        for epoch in range(self.n_epoch):
            train_metrics_epoch = self._train_step()
            train_metrics_epoch = {met_name: met_val for met_name, met_val
                                   in zip(metric_names, train_metrics_epoch)}

            self._logging(train_metrics_epoch, epoch, mode='train')

            test_metrics_epoch = self._valid_step()
            test_metrics_epoch = {met_name: met_val for met_name, met_val
                                  in zip(metric_names, test_metrics_epoch)}
            self._logging(test_metrics_epoch, epoch, mode='test')

            if epoch % self.print_every == 0:
                print_line = fill_line.format(epoch + 1,
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
        # req_grad(self.model)
        losses, n_batches = 0, 0

        y_all_pred = torch.tensor([])
        y_all_true = torch.tensor([])

        self.model.train(True)
        for x, labels in self.train_loader:

            self.optimizer.zero_grad()
            x = x.to(self.device)
            labels = labels.to(self.device)

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
        for i, (x, labels) in enumerate(self.test_loader):
            with torch.no_grad():
                x = x.to(self.device)
                labels = labels.reshape(-1, 1).to(self.device)

                y_out = self.model(x)

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
        if self.scheduler:
            self.scheduler.step()

        y_all_pred = y_all_pred.numpy().reshape([-1, 1])
        y_all_true = y_all_true.numpy().reshape([-1, 1])

        acc, pr, rec, f1 = self.calculate_metrics(y_all_true, y_all_pred)
        balance = np.sum(y_all_pred) / len(y_all_pred)
        return mean_loss, acc, pr, rec, f1, balance

    def calculate_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        pr = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        return acc, pr, rec, f1
    
    def save_metrics_as_csv(self, path):
        res = pd.DataFrame([])
        for split, metrics in self.dict_logging.items():
            df_metrics = pd.DataFrame(metrics)
            df_metrics['epoch'] = np.arange(1, len(df_metrics) + 1)
            df_metrics['split'] = split
            res = pd.concat([res, df_metrics])

        res.to_csv(path, index=False)

    def save_result(self, save_path, model_name):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        full_path = save_path + '/' + model_name
        torch.save(self.model.state_dict(), full_path + '.pth')

        self.save_metrics_as_csv(full_path+'_metrics.csv')

        # with open(full_path+'_metrics.pickle', 'wb') as f:
        #     pickle.dump(self.dict_logging, f) 
