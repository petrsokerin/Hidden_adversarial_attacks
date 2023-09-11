
from typing import Dict, Any, Tuple, List, Union, Sequence, Callable


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score




def calculate_metrics_class(y_true:np.array,
                            y_pred: np.array):
    #-> Tuple(float, float, float):
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    pr = average_precision_score(y_true, y_pred)
    return acc, roc, pr


def plot_aa_metrics(res_df, method='eps', inverse=False):
    metrics = res_df.columns[1:-1]
    if method == 'eps':
        plt.figure(figsize=(15, 12))

        for i, eps in enumerate(res_df['eps'].unique()):
            plt.subplot(2, 3, i + 1)

            for metric in metrics:
                df_loc = res_df[res_df['eps'] == eps].copy()
                data_logs = df_loc[metric]
                if inverse:
                    data_logs = 1 - data_logs

                plt.plot(df_loc.index, data_logs, label=metric)
            plt.title(f'Eps = {round(eps, 4)}')
            plt.xlabel('n iterations')
            plt.legend()
            plt.grid()
        plt.show()

    elif method == 'metric':
        plt.figure(figsize=(20, 15))

        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i + 1)

            for eps in res_df['eps'].unique():
                df_loc = res_df[res_df['eps'] == eps].copy()

                data_logs = df_loc[metric]
                if inverse:
                    data_logs = 1 - data_logs
                plt.plot(df_loc.index, data_logs, label=f'Eps = {round(eps, 4)}')

            plt.title(metric)
            plt.xlabel('n iterations')
            plt.legend()
            plt.grid()
        plt.show()
    else:
        raise ValueError('method should be eps or metric')


def get_ensemble_mi(preds: np.ndarray) -> np.ndarray:
    """Calculate entropy of a given 1d numpy array. Input values are clipped to [0, 1].

    :param p: numpy array of numerics
    :return: numpy array of entropy values
    """
    _, pentropy = get_ensemble_predictive_entropy(preds)
    ave_preds, eentropy = get_ensemble_expected_entropy(preds)

    return ave_preds, (pentropy - eentropy)


def get_minprob(pred: np.ndarray) -> np.ndarray:
    """Calculate minprob uncertainty estimate. Input values are clipped to [0, 1].
    :param pred: numpy array of numerics
    :return: numpy array of minprob estimates
    """

    def func(pred):
        if pred < 0.5:
            return pred, pred
        else:
            return pred, (1 - pred)

    func_vec = np.vectorize(func)
    clipped_pred = np.clip(pred, 1e-5, 1 - 1e-5)
    clipped_pred = np.mean(clipped_pred, axis=0)
    return func_vec(clipped_pred)


def get_entropy(p: np.ndarray) -> np.ndarray:
    """Calculate entropy of a given 1d numpy array. Input values are clipped to [0, 1].

    :param p: numpy array of numerics
    :return: numpy array of entropy values
    """
    cp = np.clip(p, 1e-5, 1 - 1e-5)
    entropy = -cp * np.log2(cp) - (1 - cp) * np.log2(1 - cp)

    return entropy


def get_ensemble_predictive_entropy(preds: np.ndarray) -> np.ndarray:
    """Calculate predictive entropy of ensemble predictions.
    :param preds: numpy array of ensemble predictions. Expects first dimension to represent members of ensemble
    :return: numpy array of predictive entropy estimates
    """
    ave_preds = np.average(np.copy(preds), axis=0)

    return ave_preds, get_entropy(ave_preds)


def get_ensemble_expected_entropy(preds: np.ndarray) -> np.ndarray:
    """Calculate expected entropy of ensemble predictions.
    :param preds: numpy array of ensemble predictions. Expects first dimension to represent members of ensemble
    :return: numpy array of predictive entropy estimates
    """
    ave_preds = np.average(np.copy(preds), axis=0)

    return ave_preds, np.apply_along_axis(get_entropy, 0, preds).mean(axis=0)


def get_ensemble_std(preds: np.ndarray) -> np.ndarray:
    """Calculate estimate of standard deviation of ensemble predictions.
    :param preds: numpy array of ensemble predictions. Expects first dimension to represent members of ensemble
    :return: numpy array of standard deviation estimates estimates
    """
    ave_preds = np.average(np.copy(preds), axis=0)

    return ave_preds, np.std(preds, axis=0)


def sort_data_by_metric(
        metric: Sequence, preds: np.ndarray, labels: np.ndarray):
    #) -> Tuple[List, List]:
    """Sort preds and labels by descending uncertainty metric.
    :param metric: uncertainty metric according to which preds and labels will be sorted
    :param preds: model predictions
    :param labels: ground truth labels
    :return: a tuple of
        - np.ndarray of predictions, sorted according to metric
        - np.ndarray of labels, sorted according to metric
    """
    sorted_metric_idx = np.argsort(metric)

    return preds[sorted_metric_idx].flatten(), labels[sorted_metric_idx].flatten()


def get_upper_bound_idx(data_len: int, rejection_rates: Sequence[float]) -> List[float]:
    """Calculate upped bounds on indices of data arrays.
    Based on corresponding list of rejection rates is applied.
    :param data_len: length of data array
    :param rejection_rates: array of rejection rates to calculate upper bounds for
    :return: list of upper bounds
    """
    idx = []
    for rate in rejection_rates:
        idx.append(
            min(np.ceil(data_len * (1 - rate)), np.array(data_len)).astype(int).item()
        )

    return idx


def reject_and_eval(
        preds: np.ndarray,
        labels: np.ndarray,
        upper_bounds: Sequence[float],
        scoring_func: Callable,
) -> List:
    """Clip preds and labels arrays.
    Using list of upper bounds, and calculate scoring metric for
    predictions after rejection.
    :param preds: model label predictions or predicted class probabilities
    :param labels: ground truth labels
    :param upper_bounds: list of upper bounds to clip preds and labels to
    :param scoring_func: scoring function that takes labels and predictions or probabilities (in that order)
    :return: list of scores calculated for each upper bound
    """
    scores = []
    predicted_labels = np.where(preds > 0.5, 1, 0)

    i = 0
    for upper_bound in upper_bounds:
        predicted_labels_below_thresh = predicted_labels[0:upper_bound]
        preds_below_thresh = preds[0:upper_bound]
        labels_below_thresh = labels[0:upper_bound]

        try:
            if preds_below_thresh.size > 0 and labels_below_thresh.mean() not in [-0.00000001, 1.000000001]:
                scores.append(scoring_func(labels_below_thresh, preds_below_thresh))

        except ValueError:
            if (
                    predicted_labels_below_thresh.size > 0
                    and labels_below_thresh.mean() not in [-0.00000001, 1.000000001]
            ):
                scores.append(
                    scoring_func(labels_below_thresh, predicted_labels_below_thresh)
                )
        i += 1

    return scores


def reject_by_metric(
        get_metric: Callable,
        preds: np.ndarray,
        labels: np.ndarray,
        rejection_rates: List[float],
        scoring_func: Callable,
) -> List:
    """Reject points from preds and labels based on uncertainty estimate of choice.
    :param get_metric: function that returns uncertainty metric for given model predictions
    :param preds: model label predictions or predicted class probabilities
    :param labels: ground truth labels
    :param rejection_rates: list of rejection rates to use
    :param scoring_func: scoring function that takes labels and predictions or probabilities (in that order)
    :return: list of scores calculated for each upper bound
    """

    preds, metric_values = get_metric(preds)

    preds_sorted, labels_sorted = sort_data_by_metric(metric_values, preds, labels)
    upper_indices = get_upper_bound_idx(preds.size, rejection_rates)

    res = reject_and_eval(preds_sorted, labels_sorted, upper_indices, scoring_func)
    return res


def reject_by_iters_obj(
        iters_vec: np.ndarray,
        preds: np.ndarray,
        labels: np.ndarray,
        rejection_rates: List[float],
        scoring_func: Callable,
) -> List:
    preds = np.average(np.copy(preds), axis=0)
    preds_sorted, labels_sorted = sort_data_by_metric(1 / (iters_vec + 0.001), preds, labels)
    # preds_sorted, labels_sorted = sort_data_by_metric(iters_vec, preds, labels)

    upper_indices = get_upper_bound_idx(preds.size, rejection_rates)
    res = reject_and_eval(preds_sorted, labels_sorted, upper_indices, scoring_func)
    return res


def reject_by_diff(
        diff_vec: np.ndarray,
        preds: np.ndarray,
        labels: np.ndarray,
        rejection_rates: List[float],
        scoring_func: Callable,
) -> List:
    preds = np.average(np.copy(preds), axis=0)

    preds_sorted, labels_sorted = sort_data_by_metric(1 / (diff_vec + 0.001), preds, labels)
    #preds_sorted, labels_sorted = sort_data_by_metric(diff_vec, preds, labels)

    upper_indices = get_upper_bound_idx(preds.size, rejection_rates)
    res = reject_and_eval(preds_sorted, labels_sorted, upper_indices, scoring_func)
    return res


def reject_by_norm(
        norm_vec: np.ndarray,
        preds: np.ndarray,
        labels: np.ndarray,
        rejection_rates: List[float],
        scoring_func: Callable,
) -> List:
    preds = np.average(np.copy(preds), axis=0)

    # print(len(iters_vec), len(preds), len(labels))
    # preds_sorted, labels_sorted = sort_data_by_metric(1/(norm_vec+1), preds, labels)
    preds_sorted, labels_sorted = sort_data_by_metric(norm_vec, preds, labels)

    upper_indices = get_upper_bound_idx(preds.size, rejection_rates)
    res = reject_and_eval(preds_sorted, labels_sorted, upper_indices, scoring_func)
    return res


def reject_randomly(
        preds: np.ndarray,
        labels: np.ndarray,
        rejection_rates: List[float],
        num_samples: int,
        scoring_func: Callable,
) -> np.ndarray:
    """Reject predictions after random shuffling.

    Perform sampling num_samples times for each rejection rate and average over them

    :param preds: model label predictions or predicted class probabilities
    :param labels: ground truth labels
    :param rejection_rates: list of rejection rates to use
    :param num_samples: number of repetitions of shuffling + rejection
    :param scoring_func: scoring function that takes labels and predictions or probabilities (in that order)
    :return: list of scores calculated for each upper bound
    """
    accs = []
    upper_indices = get_upper_bound_idx(preds.size, rejection_rates)

    for _ in range(num_samples):
        shuffle_indices = np.random.permutation(preds.size)

        rej_metrics = reject_and_eval(
            preds[shuffle_indices],
            labels[shuffle_indices],
            upper_indices,
            scoring_func,
        )
        accs.append(rej_metrics)

    return np.mean(accs, axis=0)


def build_basic_dict_curve(
        labels,
        preds: list,
        norms: np.array,
        rejection_rates: np.array,
        dict_metrics: dict,
):
    dict_curves = dict()

    preds = np.array(preds)
    ave_preds = np.average(preds, axis=0)

    for metric_name, metric in dict_metrics.items():
        dict_curve_metric = dict()

        dict_curve_metric['Predictive entropy'] = reject_by_metric(get_ensemble_predictive_entropy,
                                                                   preds,
                                                                   labels,
                                                                   rejection_rates,
                                                                   metric)

        dict_curve_metric['StD'] = reject_by_metric(get_ensemble_std,
                                                    preds,
                                                    labels,
                                                    rejection_rates,
                                                    metric)

        dict_curve_metric['MaxProb'] = reject_by_metric(get_minprob,
                                                        preds, #[0, :],
                                                        labels,
                                                        rejection_rates,
                                                        metric)

        dict_curve_metric['Random'] = reject_randomly(ave_preds,
                                                      labels,
                                                      rejection_rates,
                                                      100,
                                                      metric)

        dict_curve_metric['Grad_Norm'] = reject_by_norm(norms,
                                                        preds,
                                                        labels,
                                                        rejection_rates,
                                                        metric)

        dict_curves[metric_name] = dict_curve_metric
    return dict_curves


def build_custom_dict_curve(dict_curves: dict,
                            labels,
                            preds: list,
                            norms: np.array,
                            rejection_rates: np.array,
                            dict_metrics: dict,
                            rej_curves_dict: dict,
                            all_eps: list,
                            iter_to_break: int, ):
    for metric_name, metric in dict_metrics.items():
        dict_curve_metric = dict()

        type_ = 'diff'
        eps = all_eps[0]
        iter_ = iter_to_break
        rej_vec = abs(rej_curves_dict[eps][type_][iter_])

        dict_curve_metric[f'Iter_{type_}_eps={round(eps, 4)}_iter={iter_}'] = reject_by_diff(rej_vec,
                                                                                             preds,
                                                                                             labels,
                                                                                             rejection_rates,
                                                                                             metric)

        type_ = 'diff'
        eps = all_eps[2]
        iter_ = iter_to_break
        rej_vec = abs(rej_curves_dict[eps][type_][iter_])
        dict_curve_metric[f'Iter_{type_}_eps={round(eps, 4)}_iter={iter_}'] = reject_by_diff(rej_vec,
                                                                                             preds,
                                                                                             labels,
                                                                                             rejection_rates,
                                                                                             metric)

        type_ = 'iter_broke'
        eps = all_eps[4]
        iter_ = ''
        dict_curve_metric[f'Iter_{type_}_eps={round(eps, 4)}'] = reject_by_iters_obj(rej_curves_dict[eps][type_],
                                                                                     preds,
                                                                                     labels,
                                                                                     rejection_rates,
                                                                                     metric)

        # Union 2 dicts
        dict_curves[metric_name] = {**dict_curves[metric_name], **dict_curve_metric}

    return dict_curves


def draw_rejection_curves(dict_curves, rejection_rates, fig_size=(17, 9)):
    plt.figure(figsize=fig_size)

    for i, (metric_name, curves_dict) in enumerate(dict_curves.items()):

        plt.subplot(1, len(dict_curves), i + 1)

        for label, metric in curves_dict.items():
            if 'Iter_diff' in label:
                plt.plot(rejection_rates[0:len(metric)], metric,
                         label=label, linewidth=3, linestyle='--')
            elif 'Iter_object' in label:
                plt.plot(rejection_rates[0:len(metric)], metric,
                         label=label, linewidth=3, linestyle='-')

            elif 'Norm' in label:
                plt.plot(rejection_rates[0:len(metric)], metric,
                         label=label, linewidth=3, linestyle=':')
            else:
                plt.plot(rejection_rates[0:len(metric)], metric, label=label)
        plt.title(metric_name)
        plt.xlabel('Rejection rate')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid()

    plt.show()


def all_predict(model, loader, round_=True, multiclass=False, device='cpu'):
    y_all_pred = torch.FloatTensor([])

    for x, y_true in loader:

        x = x.to(device)
        y_true = y_true.to(device)
        y_pred = model(x)

        if round_:
            if multiclass:
                y_pred = torch.argmax(y_pred, dim=1)
            else:
                y_pred = torch.round_(y_pred)
        else:
            if multiclass:
                y_pred = torch.nn.functional.softmax(y_pred)

        y_all_pred = torch.cat((y_all_pred, y_pred.cpu().detach()), dim=0)

    if round_:
        y_all_pred = torch.round(y_all_pred)
    y_all_pred = y_all_pred.cpu().detach().numpy()
    y_all_pred = y_all_pred.reshape([-1, 1])
    return y_all_pred


def get_grad_norm(model: nn.Module,
                  loader: DataLoader,
                  criterion: nn.Module,
                  device='cpu'
                  ):
    grad_norm_model = torch.tensor(np.array([]))

    for x, y_true in loader:
        x.grad = None
        x.requires_grad = True

        x = x.to(device)
        y_true = y_true.to(device)

        y_pred = model(x)
        loss_val = criterion(y_pred, y_true)

        grad = torch.autograd.grad(loss_val, x, retain_graph=True)[0]
        grad_norm = torch.linalg.norm(grad, ord=2, dim=(1, 2)).cpu().detach()
        grad_norm_model = torch.cat((grad_norm_model, grad_norm))

    return np.array(grad_norm_model)


def rejection_curves_procedure(model: nn.Module,
                               loader: DataLoader,
                               criterion: nn.Module,
                               load_fun,
                               load_path,
                               labels,
                               device,
                               n_models=1,
                               ):
    norms_all = []
    preds_all = []

    for i, model_path in enumerate(load_path):
        # loading_weights

        model = load_fun(model=model, model_path=model_path)
        model = model.to(device)

        model.eval()

        # estimate results
        preds_round = all_predict(model=model, loader=loader,
                                  round_=True, device=device)
        metrics = calculate_metrics_class(labels, preds_round)
        acc, roc, pr_auc = metrics
        print(f"{i}th models Accuracy {acc:.3f}, ROC-AUC {roc:.3f}, PR-AUC {pr_auc:.3f}")

        preds = all_predict(model=model, loader=loader,
                            round_=False, device=device)
        preds_all.append(preds)

        model.train()
        norm = get_grad_norm(model=model, loader=loader, criterion=criterion, device=device)
        norms_all.append(norm)

    preds = np.array(preds_all).reshape(len(preds_all), preds.shape[0])
    return preds, np.sum(np.array(norms_all), axis=0)
