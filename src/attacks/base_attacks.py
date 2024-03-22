from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import get_attack
from src.estimation import AttackEstimator


class BaseAttack(ABC):
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)

    @abstractmethod
    def step(self, x, y_true):
        raise NotImplementedError("This method should be implemented by subclasses.")


class BatchIterativeAttack:
    def __init__(self, attack_name, attack_params, estimator=None):
        self.attack = get_attack(attack_name, attack_params)
        
        self.logging = bool(not estimator)
        self.estimator = estimator

        self.model = self.attack.model
        self.device = self.model.device

        if self.logging:
            self.metrics_names = AttackEstimator.get_metrics_name()
            self.metrics = pd.DataFrame(columns= self.metrics_names)

    @staticmethod
    def initialize_with_params():
        pass
        

    def log_step(self, y_true, y_pred, step_id):
        metrics_line = self.estimator.estimate(y_true, y_pred)
        metrics_line = [step_id] + list(metrics_line)
        df_line = pd.DataFrame(metrics_line, columns=self.metrics_names)
        self.metrics = pd.concat([self.metrics, df_line])


    def get_model_predictions(self, loader):
        y_pred_all_objects = torch.tensor([])  # logging predictions adversarial if realize_attack or original

        for X, y_true in loader:
            X, y_true = self.prepare_data_to_attack(X, y_true)
            y_pred = self.model(X)
            y_pred_all_objects = torch.cat((y_pred_all_objects, y_pred.cpu().detach()), dim=0)

        return y_pred_all_objects

    @staticmethod
    def prepare_data_to_attack(X, y, device):
        X.grad = None
        X.requires_grad = True

        X = X.to(device, non_blocking=True)
        y_true = y_true.to(device)
        return X, y

        
    def run_iteration_log(self):
        
        X_adv_all_objects = torch.FloatTensor([])  # logging x_adv for rebuilding dataloader
        y_true_all_objects = torch.tensor([])  # logging model for rebuilding dataloader and calculation difference with preds
        y_pred_all_objects = torch.tensor([])  # logging predictions adversarial if realize_attack or original

        for X, y_true in self.loader:
            X, y_true = self.prepare_data_to_attack(X, y_true, self.device)
            X_adv = self.attack.step(X, y_true)
            y_pred_adv = self.model(X_adv)

            X_adv_all_objects = torch.cat((X_adv_all_objects, X_adv.cpu().detach()), dim=0)
            y_true_all_objects = torch.cat((y_true_all_objects, y_true.cpu().detach()), dim=0)
            y_pred_all_objects = torch.cat((y_pred_all_objects, y_pred_adv.cpu().detach()), dim=0)

        return X_adv_all_objects, y_true_all_objects, y_pred_all_objects
    
    def run_iteration(self):
        
        X_adv_all_objects = torch.FloatTensor([])  # logging x_adv for rebuilding dataloader
        y_true_all_objects = torch.tensor([])  # logging model for rebuilding dataloader and calculation difference with preds

        for X, y_true in self.loader:
            X, y_true = self.prepare_data_to_attack(X, y_true, self.device)
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
        

    def forward(self, loader, n_steps=50):

        y_true = loader.dataset.y

        if self.logging:
            y_pred = self.get_model_predictions(loader)
            y_pred = y_pred.cpu().detach()
            self.log_step(y_true, y_pred, step_id=0)

        for step_id in tqdm(range(1, n_steps + 1)):

            if self.logging:
                X_adv, _, y_pred = self.run_iteration_log()
                self.log_step(y_true, y_pred, step_id=step_id)
            else:
                X_adv, _ = self.run_one_iter()

            loader = self.rebuild_loader(loader, X_adv, y_true)
            
    