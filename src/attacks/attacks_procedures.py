from functools import partial
from typing import Dict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm
import optuna
from optuna.trial import Trial
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.config import get_attack
from src.utils import (get_optimization_dict, update_trainer_params, 
collect_default_params)

# from .utils import req_grad #calculate_metrics_class_and_hiddens, build_df_aa_metrics)

class BatchIterativeAttack:
    def __init__(self, attack, estimator=None):
        self.attack = attack
        self.n_steps = self.attack.n_steps
        
        self.logging = bool(estimator)
        self.estimator = estimator

        self.model = self.attack.model
        self.device = self.attack.device

        if self.logging:
            print('logging')
            self.metrics_names = self.estimator.get_metrics_name()
            self.metrics = pd.DataFrame(columns= self.metrics_names)

    @staticmethod
    def initialize_with_params(
        attack_name, 
        attack_params, 
        estimator=None,
    ):

        attack = get_attack(attack_name, attack_params)
        return BatchIterativeAttack(attack, estimator)

    @staticmethod
    def initialize_with_optimization(
            train_loader,
            valid_loader,
            optuna_params,
            const_params,
    ):

        study = optuna.create_study(
            direction="maximize",
            sampler=instantiate(optuna_params["sampler"]),
            pruner=instantiate(optuna_params["pruner"]),
        )
        study.optimize(
            partial(
                BatchIterativeAttack.objective,
                params_vary=optuna_params["hyperparameters_vary"],
                const_params = const_params,
                train_loader=train_loader,
                valid_loader=valid_loader,
            ),
            n_trials=optuna_params["n_trials"],
        )
        
        default_params = collect_default_params(optuna_params["hyperparameters_vary"])
        best_params = study.best_params.copy()
        best_params = update_trainer_params(best_params, default_params)

        best_params.update(const_params)
        print("Best parameters are - %s", best_params)
        return BatchIterativeAttack.initialize_with_params(best_params)

    @staticmethod
    def objective(
        trial: Trial,
        params_vary: DictConfig,
        const_params: Dict,
        train_loader,
        valid_loader,
        optim_metric='F1'
    ) -> float:

        initial_model_parameters, _ = get_optimization_dict(params_vary, trial)
        initial_model_parameters = dict(initial_model_parameters)
        initial_model_parameters.update(const_params)

        model = BatchIterativeAttack.initialize_with_params(
            initial_model_parameters
        )
        last_epoch_metrics = model.train_model(train_loader, valid_loader)
        return last_epoch_metrics[optim_metric]


    def log_step(self, y_true, y_pred, X=None, step_id=0):
        y_true = y_true.flatten().numpy()
        y_pred = y_pred.flatten().numpy()
        y_pred_classes = np.round(y_pred)

        metrics_line = self.estimator.estimate(y_true, y_pred_classes, X)
        metrics_line = [step_id] + list(metrics_line)
        metrics_names = ['step_id'] + self.metrics_names
        df_line = pd.DataFrame(metrics_line, index=metrics_names).T
        self.metrics = pd.concat([self.metrics, df_line])


    def get_model_predictions(self, loader):
        y_pred_all_objects = torch.tensor([])  # logging predictions adversarial if realize_attack or original

        for X, y_true in loader:
            X, y_true = self.prepare_data_to_attack(X, y_true)
            y_pred = self.model(X)
            y_pred_all_objects = torch.cat((y_pred_all_objects, y_pred.cpu().detach()), dim=0)

        return y_pred_all_objects


    def prepare_data_to_attack(self, X, y):
        X.grad = None
        X.requires_grad = True

        X = X.to(self.device, non_blocking=True)
        y = y.to(self.device)
        return X, y

        
    def run_iteration_log(self, loader):
        
        X_adv_all_objects = torch.FloatTensor([])  # logging x_adv for rebuilding dataloader
        y_true_all_objects = torch.tensor([])  # logging model for rebuilding dataloader and calculation difference with preds
        y_pred_all_objects = torch.tensor([])  # logging predictions adversarial if realize_attack or original

        for X, y_true in loader:
            X, y_true = self.prepare_data_to_attack(X, y_true)
            X_adv = self.attack.step(X, y_true)
            y_pred_adv = self.model(X_adv)

            X_adv_all_objects = torch.cat((X_adv_all_objects, X_adv.cpu().detach()), dim=0)
            y_true_all_objects = torch.cat((y_true_all_objects, y_true.cpu().detach()), dim=0)
            y_pred_all_objects = torch.cat((y_pred_all_objects, y_pred_adv.cpu().detach()), dim=0)

        return X_adv_all_objects, y_true_all_objects, y_pred_all_objects
    
    def run_iteration(self, loader):
        
        X_adv_all_objects = torch.FloatTensor([])  # logging x_adv for rebuilding dataloader
        y_true_all_objects = torch.tensor([])  # logging model for rebuilding dataloader and calculation difference with preds

        for X, y_true in loader:
            X, y_true = self.prepare_data_to_attack(X, y_true)
            X_adv = self.attack.step(X, y_true)

            X_adv_all_objects = torch.cat((X_adv_all_objects, X_adv.cpu().detach()), dim=0)
            y_true_all_objects = torch.cat((y_true_all_objects, y_true.cpu().detach()), dim=0)
            
        return X_adv_all_objects, y_true_all_objects
    
    @staticmethod
    def rebuild_loader(loader, X_adv, y_true):
        dataset_class = loader.dataset.__class__
        batch_size = loader.batch_size
        dataset = dataset_class(X_adv, y_true)
        loader = DataLoader(dataset, batch_size=batch_size)
        return loader
    
    def get_metrics(self):
        return self.metrics
        
    def forward(self, loader):

        y_true = loader.dataset.y 

        if self.logging:
            y_pred = self.get_model_predictions(loader)
            y_pred = y_pred.cpu().detach()
            X = loader.dataset.X.unsqueeze(-1)
            self.log_step(y_true, y_pred, X, step_id=0)

        for step_id in tqdm(range(1, self.n_steps + 1)):

            if self.logging:
                X_adv, _, y_pred = self.run_iteration_log(loader)
                self.log_step(y_true, y_pred, X_adv, step_id=step_id)
            else:
                X_adv, _ = self.run_iteration(loader)

            loader = self.rebuild_loader(loader, X_adv, y_true)

        return X_adv


# class IterGradAttack:
#     def __init__(
#         self,
#         model,
#         loader,
#         attack_func,
#         attack_params,
#         criterion,
#         n_steps,
#         train_mode=False,
#         disc_model=None
#     ):
        
#         self.model = model
#         self.loader = loader
#         self.attack_func = attack_func
#         self.attack_params = attack_params
#         self.criterion = criterion
#         self.n_steps = n_steps
#         self.train_mode = train_mode

#         self.dataset_class = loader.dataset.__class__
#         self.device = next(model.parameters()).device
#         self.batch_size = loader.batch_size

#         self.logging = False

#         self.disc_model = disc_model

#     def run_iterations(self):

#         if self.logging:
#             x_original, y_true, preds = self.run_one_iter(realize_attack=False)
#             self.log_one_iter(0, y_true, preds, x_original)

#         for iter_ in tqdm(range(1, self.n_steps + 1)):

#             if self.logging:
#                 x_adv, y_true, preds_adv = self.run_one_iter()
#                 self.log_one_iter(iter_, y_true, preds_adv, x_adv)
#             else:
#                 x_adv, y_true = self.run_one_iter()

#             # rebuilding dataloader for new iteration
#             it_dataset = self.dataset_class(x_adv, torch.tensor(y_true))
#             self.loader = DataLoader(it_dataset, batch_size=self.batch_size)

#         return x_adv, y_true

#     def run_one_iter(self, realize_attack=True):

#         self.model.train(self.train_mode)
#         req_grad(self.model, state=False)  # detach all model's parameters

#         all_y_true = torch.tensor([])  # logging model for rebuilding dataloader and calculation difference with preds
#         x_tensor = torch.FloatTensor([])  # logging x_adv for rebuilding dataloader

#         if self.logging:
#             all_preds = []  # logging predictions adversarial if realize_attack or original

#         for x, y_true in self.loader:
#             all_y_true = torch.cat((all_y_true, y_true.cpu().detach()), dim=0)

#             x.grad = None
#             x.requires_grad = True

#             # prediction for original input
#             x = x.to(self.device, non_blocking=True)
#             y_true = y_true.to(self.device)

#             y_pred = self.model(x)

#             if realize_attack:
#                 x_adv = self.attack_func(self.model, self.criterion, x, y_true, **self.attack_params)
#                 x_tensor = torch.cat((x_tensor, x_adv.cpu().detach()), dim=0)

#                 if self.logging:
#                     with torch.no_grad():  # prediction for adv input
#                         y_pred_adv = self.model(x_adv)
#                     all_preds.extend(y_pred_adv.cpu().detach().data.numpy())
#             else:
#                 x_tensor = torch.cat((x_tensor, x.cpu().detach()), dim=0)
#                 if self.logging:
#                     all_preds.extend(y_pred.cpu().detach().data.numpy())

#         if self.logging:
#             return x_tensor.detach(), all_y_true.detach(), all_preds
#         else:
#             return x_tensor.detach(), all_y_true.detach()

#     def log_one_iter(self, iter_, y_true, preds, x):
#         if self.multiclass:
#             preds_flat_round = np.argmax(np.array(preds), axis=1).flatten()
#             shape_diff = (1, 2)
#         else:
#             preds_flat_round = np.round(np.array(preds)).flatten()
#             shape_diff = (1)

#         if iter_ == 0:
#             self.preds_no_attack = np.array(preds)

#         y_true_flat = y_true.cpu().detach().numpy().flatten()

#         mask = (preds_flat_round != y_true_flat) & (self.iter_broken_objs > iter_)
#         self.iter_broken_objs[mask] = iter_ + 1

#         self.rejection_dict['diff'][iter_ + 1] = np.sum(
#             (self.preds_no_attack - np.array(preds)) ** 2,
#             axis=shape_diff
#         )

#         self.rejection_dict['iter_broke'] = self.iter_broken_objs
#         self.aa_res_dict[iter_ + 1] = self.metric_fun(y_true_flat, preds_flat_round, x, self.disc_model)

#     def run_iterations_logging(self, metric_fun, n_objects, multiclass=False):

#         self.metric_fun = metric_fun
#         self.n_objects = n_objects
#         self.multiclass = multiclass

#         self.logging = True

#         self.aa_res_dict = dict()  # structure for saving decreasing of metrics
#         self.rejection_dict = dict()  # structure for saving rejection curves params
#         self.rejection_dict['diff'] = dict()
#         self.iter_broken_objs = np.array([10 ** 7] * n_objects)

#         self.run_iterations()

#         return self.aa_res_dict, self.rejection_dict


# def attack_procedure(
#     model: nn.Module,
#     loader: DataLoader,
#     criterion: nn.Module,
#     attack_func,
#     attack_params,
#     all_eps,
#     n_steps: int,
#     metric_func=None,
#     n_objects=100,
#     train_mode=False,
#     disc_model=None,
# ):
#     aa_res_df = pd.DataFrame()

#     rej_curves_dict = dict()  # multilevel dict  eps -> diff and object
#     # diff -> #n_iteration -> np.array difference between original prediction without attack and broken predictions
#     # object -> np.array n_iter when wrong prediction

#     for eps in tqdm(all_eps):
#         print(f'*****************  EPS={eps}  ****************')

#         attack_params['eps'] = eps
#         attack_class = IterGradAttack(model, loader, attack_func, attack_params,
#                                       criterion, n_steps, train_mode=train_mode,
#                                       disc_model=disc_model)
#         aa_res_iter_dict, rej_curves_iter_dict = attack_class.run_iterations_logging(metric_func, n_objects,
#                                                                                      multiclass=False)

#         rej_curves_dict[eps] = rej_curves_iter_dict
#         aa_res_df = pd.concat([aa_res_df, build_df_aa_metrics(aa_res_iter_dict, eps)])

#     return aa_res_df, rej_curves_dict
