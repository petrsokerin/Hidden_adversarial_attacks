from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import torch
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


class BaseEstimator(ABC):
    def __init__(self):
        self.metrics_name = []

    @abstractmethod
    def estimate(self, y_true, y_pred):
        return NotImplementedError("This method should be implemented by subclasses.")

    def get_metrics_name(self):
        return self.metrics_name
    
class AttackEstimator(BaseEstimator):
    def __init__(self):
        pass



def build_df_aa_metrics(metric_dict: dict, eps: float):
    """
    Transform dict with metrics in pd.DataFrame

    :param metric_dict: dict key iter number and values list of metrics ACC, ROC AUC, PR AUC
    :param eps: eps param to add in result df
    :return: pd.DataFrame with metrics, number of iterations and eps

    """

    results_df = pd.DataFrame.from_dict(metric_dict, orient="index")
    results_df.set_axis(
        pd.Index(["ACC", "ROC AUC", "PR AUC", "HID"], name="metric"), axis=1, inplace=True
    )
    results_df.set_axis(
        pd.Index(results_df.index, name="n steps", ), axis=0, inplace=True,
    )

    results_df = results_df.reset_index()
    results_df['eps'] = eps
    return results_df


def calculate_metrics_class(y_true: np.array,
                            y_pred: np.array):
    # -> Tuple(float, float, float):
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    pr = average_precision_score(y_true, y_pred)
    return acc, roc, pr


def calculate_hiddeness(model, X):
    model_device = next(model.parameters()).device
    X = X.to(model_device)
    hid = torch.mean(model(X))
    return hid.detach().cpu().numpy()


def calculate_metrics_class_and_hiddens(
        y_true: np.array,
        y_pred: np.array,
        X,
        disc_model=None,
):
    acc, roc, pr = calculate_metrics_class(y_true, y_pred)

    hid = calculate_hiddeness(disc_model, X) if disc_model else None

    return acc, roc, pr, hid


def calc_accuracy(model, y_pred, y_pred_adv):
    acc_val = np.mean((y_pred == model))
    acc_adv = np.mean((y_pred_adv == model))
    return acc_val, acc_adv