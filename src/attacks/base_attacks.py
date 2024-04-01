from abc import ABC, abstractmethod
from functools import partial
from typing import Dict

import optuna
from optuna.trial import Trial
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils import (get_optimization_dict, update_trainer_params, 
collect_default_params)

class BaseIterativeAttack(ABC):
    def __init__(self, model, n_steps=50, *args, **kwargs):
        self.model = model
        self.device= next(model.parameters()).device
        self.n_steps = n_steps

    @abstractmethod
    def step(self, X, y_true):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def initialize_with_params(
            self,
            params
    ):
        return self.__class__(**dict(params))

    def initialize_with_optimization(
            self,
            loader,
            optuna_params,
            const_params,
    ):
        attack_class = self.__class__
        print(attack_class)

        study = optuna.create_study(
            direction="maximize",
            sampler=instantiate(optuna_params["sampler"]),
            pruner=instantiate(optuna_params["pruner"]),
        )
        study.optimize(
            partial(
                BaseIterativeAttack.objective,
                params_vary=optuna_params["hyperparameters_vary"],
                const_params=const_params,
                loader=loader,
                optim_metric=optuna_params['optim_metric'],
                attack_class=attack_class
            ),
            n_trials=optuna_params["n_trials"],
        )
        
        default_params = collect_default_params(optuna_params["hyperparameters_vary"])
        best_params = study.best_params.copy()
        best_params = update_trainer_params(best_params, default_params)

        best_params.update(const_params)

        print("Best parameters are - %s", best_params)
        return attack_class(**dict(best_params))
    
    @staticmethod
    def objective(
        trial: Trial,
        params_vary: DictConfig,
        const_params: Dict,
        loader,
        attack_class,
        optim_metric: str = 'F_EFF_CONC',
    ) -> float:

        params = const_params

        initial_model_parameters, _ = get_optimization_dict(params_vary, trial)
        initial_model_parameters = dict(initial_model_parameters)
        params.update(initial_model_parameters)

        attack = attack_class(
            **dict(params)
        )
        attack.apply_attack(loader) 
        results = attack.get_metrics()
        last_step_metrics = results.iloc[-1]

        return last_step_metrics[optim_metric]