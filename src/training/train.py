import os
from functools import partial
from typing import Any, Dict, List
from copy import copy

import numpy as np
import optuna
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from optuna.trial import Trial
from torch.utils.data import DataLoader

from src.attacks import BaseIterativeAttack
from src.config import (
    get_attack,
    get_criterion,
    get_model,
    get_optimizer,
    get_scheduler,
)
from src.estimation import ClassifierEstimator
from src.utils import (
    collect_default_params,
    fix_seed,
    get_optimization_dict,
    update_dict_params,
)


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


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        n_epochs: int = 30,
        early_stop_patience: int = None,
        logger: Any = None,
        print_every: int = 5,
        device: str = "cpu",
        multiclass: bool = False,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.estimator = ClassifierEstimator()
        self.n_epochs = n_epochs
        self.early_stop_patience = early_stop_patience

        self.device = device
        self.multiclass = multiclass
        self.print_every = print_every

        self.logger = logger
        self.dict_logging = {}

    @staticmethod
    def initialize_with_params(
        model_name: str = "LSTM",
        model_params: Dict = None,
        criterion_name: str = "BCELoss",
        criterion_params: Dict = None,
        optimizer_name: str = "Adam",
        optimizer_params: Dict = None,
        scheduler_name: str = "None",
        scheduler_params: Dict = None,
        n_epochs: int = 30,
        early_stop_patience: int = None,
        logger: Any = None,
        print_every: int = 5,
        device: str = "cpu",
        seed: int = 0,
        multiclass: bool = False,
    ):
        fix_seed(seed)
        if model_params == "None" or not model_params:
            model_params = {}
        if criterion_params == "None" or not criterion_params:
            criterion_params = {}
        if optimizer_params == "None" or not optimizer_params:
            optimizer_params = {}
        if scheduler_params == "None" or not scheduler_params:
            scheduler_params = {}

        model = get_model(model_name, model_params, device=device)
        criterion = get_criterion(criterion_name, criterion_params)
        optimizer = get_optimizer(optimizer_name, model.parameters(), optimizer_params)
        scheduler = get_scheduler(scheduler_name, optimizer, scheduler_params)
        return Trainer(
            model,
            criterion,
            optimizer,
            scheduler,
            n_epochs=n_epochs,
            early_stop_patience=early_stop_patience,
            logger=logger,
            print_every=print_every,
            device=device,
            multiclass=multiclass,
        )

    @staticmethod
    def initialize_with_optimization(
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optuna_params: Dict,
        const_params: Dict,
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
                optim_metric=optuna_params["optim_metric"],
                const_params=const_params,
                train_loader=train_loader,
                valid_loader=valid_loader,
            ),
            n_trials=optuna_params["n_trials"],
        )

        default_params = collect_default_params(optuna_params["hyperparameters_vary"])
        print("DEFAULT", default_params)
        best_params = study.best_params.copy()
        print("BEST", best_params)
        best_params = update_dict_params(default_params, best_params)
        if "attack_params" in const_params:
            best_params["attack_params"].update(const_params["attack_params"])
        else:
            best_params.update(const_params)
        print("Best parameters are - %s", best_params)
        return Trainer.initialize_with_params(**best_params)

    @staticmethod
    def objective(
        trial: Trial,
        params_vary: DictConfig,
        optim_metric: str,
        const_params: Dict,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> float:
        initial_model_parameters, _ = get_optimization_dict(params_vary, trial)
        initial_model_parameters = dict(initial_model_parameters)
        if "attack_params" in const_params:
            initial_model_parameters["attack_params"].update(
                const_params["attack_params"]
            )
        else:
            initial_model_parameters.update(const_params)

        model = Trainer.initialize_with_params(**initial_model_parameters)
        last_epoch_metrics = model.train_model(train_loader, valid_loader)
        return last_epoch_metrics[optim_metric]

    def _logging(self, data, epoch, mode="train"):
        for metric in self.dict_logging[mode].keys():
            self.dict_logging[mode][metric].append(data[metric])

            self.logger.add_scalar(metric + "/" + mode, data[metric], epoch)

    def train_model(
        self, train_loader: DataLoader, valid_loader: DataLoader
    ) -> Dict[str, float]:
        if self.model.self_supervised:
            print("Training self-supervised model")
            X_train = train_loader.dataset.X.unsqueeze(-1).numpy()
            self.model.train_embedding(X_train, verbose=True)
            print("Training self-supervised part is finished")

        if self.early_stop_patience and self.early_stop_patience != "None":
            earl_stopper = EarlyStopper(self.early_stop_patience)

        metric_names = ["loss"] + self.estimator.get_metrics_names()
        self.dict_logging = {
            "train": {metric: [] for metric in metric_names},
            "test": {metric: [] for metric in metric_names},
        }

        fill_line = "Epoch {} train loss: {}; acc_train {}; test loss: {}; acc_test {}; f1_test {}; balance {}"

        for epoch in range(self.n_epochs):
            train_metrics_epoch = self._train_step(train_loader)
            train_metrics_epoch = {
                met_name: met_val
                for met_name, met_val in zip(metric_names, train_metrics_epoch)
            }

            self._logging(train_metrics_epoch, epoch, mode="train")

            test_metrics_epoch = self._valid_step(valid_loader)
            test_metrics_epoch = {
                met_name: met_val
                for met_name, met_val in zip(metric_names, test_metrics_epoch)
            }

            self._logging(test_metrics_epoch, epoch, mode="test")

            if epoch % self.print_every == 0:
                print_line = fill_line.format(
                    epoch + 1,
                    round(train_metrics_epoch["loss"], 3),
                    round(train_metrics_epoch["accuracy"], 3),
                    round(test_metrics_epoch["loss"], 3),
                    round(test_metrics_epoch["accuracy"], 3),
                    round(test_metrics_epoch["f1"], 3),
                    round(test_metrics_epoch["balance_pred"], 3),
                )
                print(print_line)

            if self.scheduler:
                self.scheduler.step()

            if self.early_stop_patience and self.early_stop_patience != "None":
                res_early_stop = earl_stopper.early_stop(test_metrics_epoch["loss"])
                if res_early_stop:
                    break
        return test_metrics_epoch

    def _train_step(self, loader: DataLoader) -> List[float]:
        # req_grad(self.model)
        losses = 0

        y_all_pred = torch.tensor([])
        y_all_true = torch.tensor([])

        self.model.train(True)
        for x, labels in loader:
            self.optimizer.zero_grad()
            x = x.to(self.device)
            labels = labels.to(self.device)

            y_out = self.model(x)

            loss = self.criterion(y_out, labels)

            loss.backward()
            self.optimizer.step()
            losses += loss

            if self.multiclass:
                y_pred = torch.argmax(y_out, axis=1)
            else:
                y_pred = torch.round(y_out)

            y_all_true = torch.cat((y_all_true, labels.cpu().detach()), dim=0)
            y_all_pred = torch.cat((y_all_pred, y_pred.cpu().detach()), dim=0)

        mean_loss = losses.cpu().detach().numpy() / len(loader)

        y_all_pred = y_all_pred.numpy().reshape([-1, 1])
        y_all_true = y_all_true.numpy().reshape([-1, 1])

        metrics = self.estimator.estimate(y_all_true, y_all_pred)

        metrics = [mean_loss] + metrics
        return metrics

    def _valid_step(self, loader: DataLoader) -> List[float]:
        y_all_pred = torch.tensor([])
        y_all_true = torch.tensor([])

        losses = 0
        self.model.eval()
        for x, labels in loader:
            with torch.no_grad():
                x = x.to(self.device)
                labels = labels.reshape(-1, 1).to(self.device)

                y_out = self.model(x)

                loss = self.criterion(y_out, labels)
                losses += loss

                if self.multiclass:
                    y_pred = torch.argmax(y_out, axis=1)
                else:
                    y_pred = torch.round(y_out)

            y_all_true = torch.cat((y_all_true, labels.cpu().detach()), dim=0)
            y_all_pred = torch.cat((y_all_pred, y_pred.cpu().detach()), dim=0)

        mean_loss = losses.cpu().detach().numpy() / len(loader)

        y_all_pred = y_all_pred.numpy().reshape([-1, 1])
        y_all_true = y_all_true.numpy().reshape([-1, 1])

        metrics = self.estimator.estimate(y_all_true, y_all_pred)
        metrics = [mean_loss] + metrics
        return metrics

    def save_metrics_as_csv(self, path: str) -> None:
        res = pd.DataFrame([])
        for split, metrics in self.dict_logging.items():
            df_metrics = pd.DataFrame(metrics)
            df_metrics["epoch"] = np.arange(1, len(df_metrics) + 1)
            df_metrics["split"] = split
            res = pd.concat([res, df_metrics])

        res.to_csv(path, index=False)

    def save_result(self, save_path: str, model_name: str) -> None:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        full_path = save_path + "/" + model_name
        torch.save(self.model.state_dict(), full_path + ".pth")

        self.save_metrics_as_csv(full_path + "_metrics.csv")


class DiscTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        attack: BaseIterativeAttack,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        n_epochs: int = 30,
        early_stop_patience: int = None,
        logger: Any = None,
        print_every: int = 5,
        device: str = "cpu",
        multiclass: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=n_epochs,
            early_stop_patience=early_stop_patience,
            logger=logger,
            print_every=print_every,
            device=device,
            multiclass=multiclass,
        )

        self.attack = attack

    @staticmethod
    def initialize_with_params(
        model_name: str = "LSTM",
        model_params: Dict = None,
        attack_name: str = "FGSM",
        attack_params: Dict = None,
        criterion_name: str = "BCELoss",
        criterion_params: Dict = None,
        optimizer_name: str = "Adam",
        optimizer_params: Dict = None,
        scheduler_name: str = "None",
        scheduler_params: Dict = None,
        n_epochs: int = 30,
        early_stop_patience: int = None,
        logger: Any = None,
        print_every: int = 5,
        device: str = "cpu",
        seed: int = 0,
        multiclass: bool = False,
    ):
        fix_seed(seed)
        if model_params == "None" or not model_params:
            model_params = {}
        if criterion_params == "None" or not criterion_params:
            criterion_params = {}
        if optimizer_params == "None" or not optimizer_params:
            optimizer_params = {}
        if scheduler_params == "None" or not scheduler_params:
            scheduler_params = {}

        model = get_model(model_name, model_params, device=device)
        criterion = get_criterion(criterion_name, criterion_params)
        optimizer = get_optimizer(optimizer_name, model.parameters(), optimizer_params)
        scheduler = get_scheduler(scheduler_name, optimizer, scheduler_params)

        attack = get_attack(attack_name, attack_params)

        return DiscTrainer(
            model=model,
            attack=attack,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=n_epochs,
            early_stop_patience=early_stop_patience,
            logger=logger,
            print_every=print_every,
            device=device,
            multiclass=multiclass,
        )
    
    @staticmethod
    def initialize_with_optimization(
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optuna_params: Dict,
        const_params: Dict,
    ):
        study = optuna.create_study(
            direction="maximize",
            sampler=instantiate(optuna_params["sampler"]),
            pruner=instantiate(optuna_params["pruner"]),
        )
        study.optimize(
            partial(
                DiscTrainer.objective,
                params_vary=optuna_params["hyperparameters_vary"],
                optim_metric=optuna_params["optim_metric"],
                const_params=copy(const_params),
                train_loader=train_loader,
                valid_loader=valid_loader,
            ),
            n_trials=optuna_params["n_trials"],
        )

        default_params = collect_default_params(optuna_params["hyperparameters_vary"])
        print("DEFAULT", default_params)
        best_params = study.best_params.copy()
        print("BEST", best_params)
        best_params = update_dict_params(default_params, best_params)
        if "attack_params" in const_params:
            new_attack_params = set(const_params["attack_params"]) - set(best_params["attack_params"])
            for param in new_attack_params:
                best_params["attack_params"][param] = const_params["attack_params"][param]
            del const_
        best_params.update(const_params)
        print("Best parameters are - %s", best_params)
        return DiscTrainer.initialize_with_params(**best_params)

    @staticmethod
    def objective(
        trial: Trial,
        params_vary: DictConfig,
        optim_metric: str,
        const_params: Dict,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> float:
        initial_model_parameters, _ = get_optimization_dict(params_vary, trial)
        initial_model_parameters = dict(initial_model_parameters)
        if "attack_params" in const_params:
            new_attack_params = set(const_params["attack_params"]) - set(initial_model_parameters["attack_params"])
            for param in new_attack_params:
                initial_model_parameters["attack_params"][param] = const_params["attack_params"][param]
            del const_params['attack_params']
        initial_model_parameters.update(const_params)

        print(initial_model_parameters)

        model = DiscTrainer.initialize_with_params(**initial_model_parameters)
        last_epoch_metrics = model.train_model(train_loader, valid_loader)
        return last_epoch_metrics[optim_metric]

    def _generate_adversarial_data(self, loader: DataLoader) -> DataLoader:
        X_orig = torch.tensor(loader.dataset.X)
        X_adv = self.attack.apply_attack(loader).squeeze(-1)

        assert X_orig.shape == X_adv.shape

        disc_labels_zeros = torch.zeros_like(loader.dataset.y)
        disc_labels_ones = torch.ones_like(loader.dataset.y)

        new_x = torch.concat([X_orig, X_adv], dim=0)
        new_y = torch.concat([disc_labels_zeros, disc_labels_ones], dim=0)

        dataset_class = loader.dataset.__class__
        dataset = dataset_class(new_x, new_y)

        loader = DataLoader(dataset, batch_size=loader.batch_size, shuffle=True)

        return loader

    def train_model(
        self, train_loader: DataLoader, valid_loader: DataLoader
    ) -> Dict[str, float]:
        train_loader = self._generate_adversarial_data(train_loader)
        valid_loader = self._generate_adversarial_data(valid_loader)

        return super().train_model(train_loader, valid_loader)
