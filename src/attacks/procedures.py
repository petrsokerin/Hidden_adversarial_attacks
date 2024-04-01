from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.estimation import BaseEstimator


class BatchIterativeAttack:
    def __init__(self, estimator: BaseEstimator = None, *args, **kwargs) -> None:
        self.logging = bool(estimator)
        self.estimator = estimator

        if self.logging:
            print("logging")
            self.metrics_names = self.estimator.get_metrics_names()
            self.metrics = pd.DataFrame(columns=self.metrics_names)

    def log_step(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        X: torch.Tensor = None,
        step_id: int = 0,
    ) -> None:
        y_true = y_true.flatten().numpy()
        y_pred = y_pred.flatten().numpy()
        y_pred_classes = np.round(y_pred)

        metrics_line = self.estimator.estimate(y_true, y_pred_classes, X)
        metrics_line = [step_id] + list(metrics_line)
        metrics_names = ["step_id"] + self.metrics_names
        df_line = pd.DataFrame(metrics_line, index=metrics_names).T
        self.metrics = pd.concat([self.metrics, df_line])

    def get_model_predictions(self, loader: DataLoader) -> torch.Tensor:
        y_pred_all_objects = torch.tensor(
            []
        )  # logging predictions adversarial if realize_attack or original

        for X, y_true in loader:
            X, y_true = self.prepare_data_to_attack(X, y_true)
            y_pred = self.model(X)
            y_pred_all_objects = torch.cat(
                (y_pred_all_objects, y_pred.cpu().detach()), dim=0
            )

        return y_pred_all_objects

    def prepare_data_to_attack(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[
        torch.Tensor
    ]:
        X.grad = None
        X.requires_grad = True

        X = X.to(self.device, non_blocking=True)
        y = y.to(self.device)
        return X, y

    def run_iteration_log(self, loader: DataLoader) -> Tuple[torch.Tensor]:
        X_adv_all_objects = torch.FloatTensor(
            []
        )  # logging x_adv for rebuilding dataloader
        y_true_all_objects = torch.tensor(
            []
        )  # logging model for rebuilding dataloader and calculation difference with preds
        y_pred_all_objects = torch.tensor(
            []
        )  # logging predictions adversarial if realize_attack or original

        for X, y_true in loader:
            X, y_true = self.prepare_data_to_attack(X, y_true)
            X_adv = self.step(X, y_true)
            y_pred_adv = self.model(X_adv)

            X_adv_all_objects = torch.cat(
                (X_adv_all_objects, X_adv.cpu().detach()), dim=0
            )
            y_true_all_objects = torch.cat(
                (y_true_all_objects, y_true.cpu().detach()), dim=0
            )
            y_pred_all_objects = torch.cat(
                (y_pred_all_objects, y_pred_adv.cpu().detach()), dim=0
            )

        return X_adv_all_objects, y_true_all_objects, y_pred_all_objects

    def run_iteration(self, loader: DataLoader) -> Tuple[torch.Tensor]:
        X_adv_all_objects = torch.FloatTensor(
            []
        )  # logging x_adv for rebuilding dataloader
        y_true_all_objects = torch.tensor(
            []
        )  # logging model for rebuilding dataloader and calculation difference with preds

        for X, y_true in loader:
            X, y_true = self.prepare_data_to_attack(X, y_true)
            X_adv = self.step(X, y_true)

            X_adv_all_objects = torch.cat(
                (X_adv_all_objects, X_adv.cpu().detach()), dim=0
            )
            y_true_all_objects = torch.cat(
                (y_true_all_objects, y_true.cpu().detach()), dim=0
            )

        return X_adv_all_objects, y_true_all_objects

    @staticmethod
    def rebuild_loader(loader, X_adv: torch.Tensor, y_true: torch.Tensor) -> DataLoader:
        dataset_class = loader.dataset.__class__
        batch_size = loader.batch_size
        dataset = dataset_class(X_adv, y_true)
        loader = DataLoader(dataset, batch_size=batch_size)
        return loader

    def get_metrics(self) -> pd.DataFrame:
        return self.metrics

    def apply_attack(self, loader: DataLoader) -> torch.Tensor:
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
