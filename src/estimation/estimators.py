from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import torch
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score


class BaseEstimator(ABC):
    def __init__(self):
        self.metrics_name = []

    @abstractmethod
    def estimate(self, y_true, y_pred):
        return NotImplementedError("This method should be implemented by subclasses.")

    def get_metrics_name(self):
        return self.metrics_name


class ClassifierEstimator(BaseEstimator):
    def __init__(self):
        self.metrics = {
            'accuracy': accuracy_score,
            'precision': roc_auc_score,
            'recall': average_precision_score,
            'f1': f1_score,
            'balance_true': lambda y_true, y_pred: np.mean(y_true),
            'balance_pred': lambda y_true, y_pred: np.mean(y_pred),
        }
        self.metrics_name = list(self.metrics.keys())

    def estimate(self, y_true, y_pred):
        metrics_res = []
        for _, metric_func in self.metrics.items():
            metrics_res.append(metric_func(y_true, y_pred))
        return metrics_res


class AttackEstimator(BaseEstimator):
    def __init__(self, disc_models=None, metric_effect='F1'):
        self.metrics = {
            'ACC': accuracy_score,
            'ROC': roc_auc_score,
            'PR': average_precision_score,
            'F1': f1_score,
        }
        self.metric_effect = metric_effect
        
        self.metrics_name = list(self.metrics.keys()) + ['EFF']
        
        self.calculate_hid = bool(disc_models)
        if disc_models:
            self.disc_model = disc_models
            self.metrics_name += ['HID', 'CONC']

    def calculate_effectiveness(self, y_true, y_pred):
        metric_res = {}

        for metric_name, metric_func in self.metrics.items():
            metric_res[metric_name] = metric_func(y_true, y_pred)

        metric_res['EFF'] = 1 - metric_res[self.metric_effect]
        return list(metric_res.values())


    def calculate_hiddeness(self, X):
        model_device = next(self.disc_models[0].parameters()).device
        X = X.to(model_device)

        hid_list = list()
        for disc_model in self.disc_models:
            hid = torch.mean(disc_model(X)).detach().cpu().numpy()
            hid_list.append(hid)
        
        hid = max(hid_list)
        conc = 1 - hid
        
        return list(hid, conc)

    
    def estimate(self, y_true, y_pred, X=None):
        metrics = self.calculate_effectiveness(y_true, y_pred)

        if self.calculate_hid:
            metric_hid = self.calculate_hiddeness(X)
            metrics = metrics + metric_hid
        return metrics

# def calculate_metrics_class_and_hiddens(
#         y_true: np.array,
#         y_pred: np.array,
#         X,
#         disc_model=None,
# ):
#     acc, roc, pr = calculate_metrics_class(y_true, y_pred)

#     hid = calculate_hiddeness(disc_model, X) if disc_model else None

#     return acc, roc, pr, hid



# def build_df_aa_metrics(metric_dict: dict, eps: float):
#     """
#     Transform dict with metrics in pd.DataFrame

#     :param metric_dict: dict key iter number and values list of metrics ACC, ROC AUC, PR AUC
#     :param eps: eps param to add in result df
#     :return: pd.DataFrame with metrics, number of iterations and eps

#     """

#     results_df = pd.DataFrame.from_dict(metric_dict, orient="index")
#     results_df.set_axis(
#         pd.Index(["ACC", "ROC AUC", "PR AUC", "HID"], name="metric"), axis=1, inplace=True
#     )
#     results_df.set_axis(
#         pd.Index(results_df.index, name="n steps", ), axis=0, inplace=True,
#     )

#     results_df = results_df.reset_index()
#     results_df['eps'] = eps
#     return results_df


# def calculate_metrics_class(y_true: np.array,
#                             y_pred: np.array):
#     # -> Tuple(float, float, float):
#     acc = accuracy_score(y_true, y_pred)
#     roc = roc_auc_score(y_true, y_pred)
#     pr = average_precision_score(y_true, y_pred)
#     return acc, roc, pr


# def calculate_hiddeness(model, X):
#     model_device = next(model.parameters()).device
#     X = X.to(model_device)
#     hid = torch.mean(model(X))
#     return hid.detach().cpu().numpy()


# def calculate_metrics_class_and_hiddens(
#         y_true: np.array,
#         y_pred: np.array,
#         X,
#         disc_model=None,
# ):
#     acc, roc, pr = calculate_metrics_class(y_true, y_pred)

#     hid = calculate_hiddeness(disc_model, X) if disc_model else None

#     return acc, roc, pr, hid


# def calc_accuracy(model, y_pred, y_pred_adv):
#     acc_val = np.mean((y_pred == model))
#     acc_adv = np.mean((y_pred_adv == model))
#     return acc_val, acc_adv