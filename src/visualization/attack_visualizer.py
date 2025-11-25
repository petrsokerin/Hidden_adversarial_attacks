import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _prepare_array(tensor) -> np.ndarray:
    if tensor is None:
        return None
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    return tensor.numpy()


def save_attack_visualizations(
    save_dir: str,
    X_orig,
    X_adv,
    y_true,
    y_pred_orig,
    y_pred_adv,
    max_samples: int = 5,
) -> None:
    """
    Save comparison plots for original vs adversarial samples and classifier outputs.
    """
    os.makedirs(save_dir, exist_ok=True)

    X_orig_np = _prepare_array(X_orig)
    X_adv_np = _prepare_array(X_adv)
    y_true_np = _prepare_array(y_true).flatten()
    y_pred_orig_np = _prepare_array(y_pred_orig).flatten()
    y_pred_adv_np = _prepare_array(y_pred_adv).flatten()

    # Ensure time dimension exists
    if X_orig_np.ndim == 2:
        X_orig_np = X_orig_np[..., np.newaxis]
    if X_adv_np.ndim == 2:
        X_adv_np = X_adv_np[..., np.newaxis]

    sequence_len = X_orig_np.shape[1]
    time_axis = np.arange(sequence_len)

    # Plot a few sample time series
    n_samples = min(max_samples, X_orig_np.shape[0])
    for idx in range(n_samples):
        plt.figure(figsize=(10, 4))
        plt.plot(
            time_axis,
            X_orig_np[idx, :, 0],
            label="Original",
            linewidth=1.5,
        )
        plt.plot(
            time_axis,
            X_adv_np[idx, :, 0],
            label="Adversarial",
            linewidth=1.2,
            alpha=0.8,
        )
        plt.title(f"Sample {idx}: Original vs Adversarial")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{idx}_timeseries.png"))
        plt.close()

    plt.figure(figsize=(8, 4))
    idx_axis = np.arange(len(y_pred_orig_np))
    plt.scatter(
        idx_axis,
        y_pred_orig_np,
        label="Original predictions",
        alpha=0.6,
        s=15,
    )
    plt.scatter(
        idx_axis,
        y_pred_adv_np,
        label="Adversarial predictions",
        alpha=0.6,
        s=15,
    )
    plt.xlabel("Sample index")
    plt.ylabel("Predicted probability")
    plt.title("Classifier outputs before/after attack")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_scatter.png"))
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(
        y_pred_orig_np,
        y_pred_adv_np,
        alpha=0.4,
        s=15,
        c=y_true_np,
        cmap="coolwarm",
    )
    plt.xlabel("Original prediction")
    plt.ylabel("Adversarial prediction")
    plt.title("Prediction shift due to attack")
    plt.colorbar(label="True label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_shift_scatter.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(
        y_pred_orig_np,
        bins=40,
        alpha=0.6,
        label="Original",
    )
    plt.hist(
        y_pred_adv_np,
        bins=40,
        alpha=0.6,
        label="Adversarial",
    )
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Distribution of classifier outputs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_distribution_hist.png"))
    plt.close()
