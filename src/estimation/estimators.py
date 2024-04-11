from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


class BaseEstimator(ABC):
    def __init__(self) -> None:
        self.metrics_names = []

    @abstractmethod
    def estimate(self, y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
        return NotImplementedError("This method should be implemented by subclasses.")

    def get_metrics_names(self) -> List[str]:
        return self.metrics_names


class ClassifierEstimator(BaseEstimator):
    def __init__(self) -> None:
        self.metrics = {
            "accuracy": accuracy_score,
            "precision": roc_auc_score,
            "recall": average_precision_score,
            "f1": f1_score,
            "balance_true": lambda y_true, y_pred: np.mean(y_true).item(),
            "balance_pred": lambda y_true, y_pred: np.mean(y_pred).item(),
        }
        self.metrics_names = list(self.metrics.keys())

    def estimate(self, y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
        metrics_res = []
        for _, metric_func in self.metrics.items():
            metrics_res.append(metric_func(y_true, y_pred))
        return metrics_res


class AttackEstimator(BaseEstimator):
    def __init__(
        self, disc_models: List[torch.nn.Module] = None, metric_effect: str = "F1"
    ) -> None:
        self.metrics = {
            "ACC": accuracy_score,
            "ROC": roc_auc_score,
            "PR": average_precision_score,
            "F1": f1_score,
        }
        self.metric_effect = metric_effect

        self.metrics_names = list(self.metrics.keys()) + ["EFF", "L1", "ACC_ORIG_ADV"]

        self.calculate_hid = bool(disc_models)
        if disc_models:
            self.disc_models = disc_models
            self.metrics_names += ["HID", "CONC", "F_EFF_CONC"]

    def calculate_effectiveness(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> List[float]:
        metric_res = {}

        for metric_name, metric_func in self.metrics.items():
            metric_res[metric_name] = metric_func(y_true, y_pred)

        metric_res["EFF"] = 1 - metric_res[self.metric_effect]
        return metric_res

    def calculate_hiddeness(self, X: np.ndarray) -> Dict[str, float]:
        model_device = next(self.disc_models[0].parameters()).device
        X = torch.tensor(X).to(model_device)

        hid_list = list()
        for disc_model in self.disc_models:
            hid = torch.mean(disc_model(X)).detach().cpu().numpy()
            hid_list.append(hid)

        hid = max(hid_list)
        conc = 1 - hid
        return {"HID": hid, "CONC": conc}

    @staticmethod
    def calculate_l1(X_orig: np.ndarray, X_adv: np.ndarray) -> float:
        data_shape_no_ax0 = tuple(np.arange(1, len(X_adv.shape)))
        l1_vector = np.sum(np.abs(X_orig - X_adv), axis=data_shape_no_ax0)
        assert l1_vector.shape[0] == len(X_orig)
        l1 = np.mean(l1_vector)
        return l1.item()

    @staticmethod
    def calculate_f_eff_conc(effectiveness: float, concealability: float) -> float:
        return 2 * effectiveness * concealability / (effectiveness + concealability)

    def estimate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_orig: np.ndarray,
        X_orig: np.ndarray,
        X_adv: np.ndarray,
    ) -> List[float]:
        metrics = self.calculate_effectiveness(y_true, y_pred)

        metrics["L1"] = self.calculate_l1(X_orig, X_adv)
        metrics["ACC_ORIG_ADV"] = accuracy_score(y_pred_orig, y_pred)

        if self.calculate_hid:
            metric_hid = self.calculate_hiddeness(X_adv)
            metric_hid["F_EFF_CONC"] = self.calculate_f_eff_conc(
                metrics["EFF"], metric_hid["CONC"]
            )
            metrics.update(metric_hid)

        return_order_metrics = [metrics[name] for name in self.metrics_names]
        return return_order_metrics
