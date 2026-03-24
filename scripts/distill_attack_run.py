import argparse
import csv
import math
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks.fgsm import FGSMAttack
from src.attacks.trainable_attack import TrainAttack
from src.config import get_model
from src.data import MyDataset, load_data, transform_data
from src.estimation.estimators import AttackEstimator
from src.utils import fix_seed, req_grad


class DistillationDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, teacher_adv: torch.Tensor) -> None:
        self.X = X
        self.y = y
        self.teacher_adv = teacher_adv

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        teacher_adv = torch.tensor(self.teacher_adv[idx], dtype=torch.float32)
        if len(X.shape) == 1:
            X = X.unsqueeze(-1)
        if len(teacher_adv.shape) == 1:
            teacher_adv = teacher_adv.unsqueeze(-1)
        return X, y, teacher_adv


def choose_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested_device)
    return torch.device("cpu")


def dataset_exists(dataset_name: str) -> bool:
    ucr_dir = ROOT / "data" / "UCR" / dataset_name
    uea_dir = ROOT / "data" / "UEA" / dataset_name
    return ucr_dir.is_dir() or uea_dir.is_dir()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path, dataset_cfg: Dict = None) -> Dict:
    cfg = OmegaConf.load(path)
    base = OmegaConf.create({})
    if dataset_cfg:
        base = OmegaConf.create({"dataset": dataset_cfg})
    merged = OmegaConf.merge(base, cfg)
    return OmegaConf.to_container(merged, resolve=True)


def load_dataset_cfg(dataset_name: str) -> Dict:
    return load_yaml(ROOT / "config_examples" / "dataset" / f"{dataset_name}.yaml")


def load_model_cfg(model_name: str, dataset_cfg: Dict) -> Dict:
    return load_yaml(ROOT / "config_examples" / "model" / f"{model_name}.yaml", dataset_cfg)


def get_classifier_criterion(n_classes: int) -> torch.nn.Module:
    if n_classes > 2:
        return torch.nn.CrossEntropyLoss()
    return torch.nn.BCELoss()


def classifier_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, n_classes: int) -> Dict[str, float]:
    labels_np = labels.cpu().numpy().reshape(-1)
    if n_classes > 2:
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        f1 = f1_score(labels_np, preds, average="macro")
    else:
        probs = logits.detach().cpu().numpy().reshape(-1)
        preds = (probs >= 0.5).astype(int)
        f1 = f1_score(labels_np, preds, zero_division=0)
    acc = accuracy_score(labels_np, preds)
    return {"accuracy": float(acc), "f1": float(f1)}


def run_classifier_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = torch.device("cpu"),
    n_classes: int = 2,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    logits_all = []
    labels_all = []
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        if train_mode:
            optimizer.zero_grad()
        logits = model(X)
        labels = y.squeeze(-1).long() if n_classes > 2 else y
        loss = criterion(logits, labels)
        if train_mode:
            loss.backward()
            optimizer.step()
        total_loss += loss.detach().item()
        logits_all.append(logits.detach().cpu())
        labels_all.append(y.detach().cpu())

    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    metrics = classifier_metrics_from_logits(logits_all, labels_all, n_classes)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def save_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def safe_cuda_synchronize():
    if torch.cuda.is_available():
        yield
        return

    original_sync = torch.cuda.synchronize
    torch.cuda.synchronize = lambda: None
    try:
        yield
    finally:
        torch.cuda.synchronize = original_sync


def ensure_sequence_channel_dim(X: torch.Tensor) -> torch.Tensor:
    if len(X.shape) == 2:
        return X.unsqueeze(-1)
    return X


def compute_attack_metrics_from_tensors(
    target_model: torch.nn.Module,
    X: torch.Tensor,
    X_adv: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    metric_effect: str = "F1",
    metric_hid: str = "ROC_AUC_DISC",
) -> Dict[str, float]:
    device = next(target_model.parameters()).device
    target_model.eval()
    X = ensure_sequence_channel_dim(X)
    X_adv = ensure_sequence_channel_dim(X_adv)
    with torch.no_grad():
        logits_orig = target_model(X.to(device))
        logits_adv = target_model(X_adv.to(device))

    y_true = y.view(-1).cpu().numpy()
    if n_classes > 2:
        y_pred_orig_prob = logits_orig.detach().cpu().numpy()
        y_pred_adv_prob = logits_adv.detach().cpu().numpy()
        y_pred_orig_classes = logits_orig.argmax(dim=-1).cpu().numpy()
        y_pred_adv_classes = logits_adv.argmax(dim=-1).cpu().numpy()
    else:
        y_pred_orig_prob = logits_orig.detach().cpu().view(-1).numpy()
        y_pred_adv_prob = logits_adv.detach().cpu().view(-1).numpy()
        y_pred_orig_classes = np.round(y_pred_orig_prob)
        y_pred_adv_classes = np.round(y_pred_adv_prob)

    estimator = AttackEstimator(
        disc_models=None,
        metric_effect=metric_effect,
        metric_hid=metric_hid,
        batch_size=None,
        n_classes=n_classes,
    )
    metrics_line = estimator.estimate(
        y_true=y_true,
        y_pred=y_pred_adv_prob,
        y_pred_classes=y_pred_adv_classes,
        y_pred_orig=y_pred_orig_classes,
        X_orig=X.detach().cpu().numpy(),
        X_adv=X_adv.detach().cpu().numpy(),
        step_id=1,
        elapsed_time=0.0,
    )
    metrics = dict(zip(estimator.metrics_names, metrics_line))
    metrics["clean_acc"] = float(accuracy_score(y_true, y_pred_orig_classes))
    return metrics


def evaluate_attack_outputs(
    target_model: torch.nn.Module,
    X: torch.Tensor,
    X_adv: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
) -> Dict[str, float]:
    attack_metrics = compute_attack_metrics_from_tensors(
        target_model=target_model,
        X=X,
        X_adv=X_adv,
        y=y,
        n_classes=n_classes,
    )
    return {
        "clean_acc": attack_metrics["clean_acc"],
        "acc": attack_metrics["ACC"],
        "fooling_rate": attack_metrics["FR"],
        "eff": attack_metrics["EFF"],
        "f1": attack_metrics["F1"],
        "raw_metrics": attack_metrics,
    }


def train_classifier(
    model_name: str,
    dataset_cfg: Dict,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.nn.Module, Path, Dict[str, float]]:
    ckpt_dir = ROOT / "checkpoints" / args.dataset / f"{model_name}_seed={args.seed}" / args.exp_name
    model_path = ckpt_dir / f"model_{model_name}_{args.seed}_{args.dataset}.pt"
    metrics_path = ckpt_dir / f"model_{model_name}_{args.seed}_{args.dataset}_metrics.csv"
    summary_path = ckpt_dir / f"model_{model_name}_{args.seed}_{args.dataset}_summary.yaml"

    model_cfg = load_model_cfg(model_name, dataset_cfg)
    model = get_model(model_name, model_cfg["params"], device=str(device))

    if model_path.exists() and not args.force_classifiers:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        valid_metrics = run_classifier_epoch(
            model=model,
            loader=valid_loader,
            criterion=get_classifier_criterion(dataset_cfg["num_classes"]),
            optimizer=None,
            device=device,
            n_classes=dataset_cfg["num_classes"],
        )
        return model, model_path, valid_metrics

    criterion = get_classifier_criterion(dataset_cfg["num_classes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.classifier_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.classifier_step_size,
        gamma=args.classifier_gamma,
    )

    history = []
    best_state = deepcopy(model.state_dict())
    best_val_loss = math.inf
    patience = 0

    for epoch in range(1, args.classifier_epochs + 1):
        train_metrics = run_classifier_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            n_classes=dataset_cfg["num_classes"],
        )
        valid_metrics = run_classifier_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            n_classes=dataset_cfg["num_classes"],
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_f1": valid_metrics["f1"],
        }
        history.append(row)
        print(
            f"[classifier:{model_name}] epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} valid_loss={valid_metrics['loss']:.4f} "
            f"valid_acc={valid_metrics['accuracy']:.4f} valid_f1={valid_metrics['f1']:.4f}"
        )

        if valid_metrics["loss"] < best_val_loss:
            best_val_loss = valid_metrics["loss"]
            best_state = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.classifier_patience:
                break

    ensure_dir(ckpt_dir)
    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), model_path)
    save_csv(metrics_path, history)
    with summary_path.open("w") as handle:
        yaml.safe_dump(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "model_name": model_name,
                "model_path": str(model_path),
                "best_valid_loss": float(best_val_loss),
            },
            handle,
            sort_keys=False,
        )
    final_metrics = run_classifier_epoch(
        model=model,
        loader=valid_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        n_classes=dataset_cfg["num_classes"],
    )
    return model, model_path, final_metrics


def rollout_ifgsm(
    model: torch.nn.Module,
    loader: DataLoader,
    eps: float,
    n_steps: int,
    n_classes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    attack = FGSMAttack(
        model=model,
        criterion=get_classifier_criterion(n_classes),
        estimator=None,
        eps=eps,
        n_steps=n_steps,
        n_classes=n_classes,
    )
    model.eval()
    with safe_cuda_synchronize():
        X_adv = attack.apply_attack(loader)
    labels = loader.dataset.y.detach().cpu()
    return X_adv.detach().cpu(), labels


def save_teacher_artifacts(
    target_model_name: str,
    split_name: str,
    teacher_adv: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> Path:
    teacher_dir = ROOT / "artifacts" / "ifgsm_teacher" / args.dataset / f"{target_model_name}_seed={args.seed}"
    ensure_dir(teacher_dir)
    teacher_path = teacher_dir / f"{split_name}_eps={args.teacher_eps}_steps={args.teacher_steps}.pt"
    torch.save(
        {
            "X_adv": teacher_adv,
            "y": labels,
            "eps": args.teacher_eps,
            "n_steps": args.teacher_steps,
            "dataset": args.dataset,
            "target_model": target_model_name,
            "seed": args.seed,
        },
        teacher_path,
    )
    return teacher_path


def generate_or_load_teacher(
    model: torch.nn.Module,
    target_model_name: str,
    split_name: str,
    loader: DataLoader,
    args: argparse.Namespace,
    dataset_cfg: Dict,
    device: torch.device,
) -> Tuple[torch.Tensor, Path]:
    teacher_dir = ROOT / "artifacts" / "ifgsm_teacher" / args.dataset / f"{target_model_name}_seed={args.seed}"
    teacher_path = teacher_dir / f"{split_name}_eps={args.teacher_eps}_steps={args.teacher_steps}.pt"
    if teacher_path.exists() and not args.force_teachers:
        payload = torch.load(teacher_path, map_location="cpu")
        return payload["X_adv"], teacher_path

    teacher_adv, labels = rollout_ifgsm(
        model=model,
        loader=loader,
        eps=args.teacher_eps,
        n_steps=args.teacher_steps,
        n_classes=dataset_cfg["num_classes"],
        device=device,
    )
    return teacher_adv, save_teacher_artifacts(target_model_name, split_name, teacher_adv, labels, args)


def merge_dicts(base: Dict, update: Dict) -> Dict:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_gen_recipe(
    gen_model_name: str,
    target_model_name: str,
    dataset_name: str,
    dataset_cfg: Dict,
) -> Tuple[Dict, Dict]:
    model_cfg = load_model_cfg(gen_model_name, dataset_cfg)
    gen_params = deepcopy(model_cfg["params"])
    train_recipe = {}

    recipe_path = (
        ROOT
        / "config_examples"
        / "hparams"
        / "TrainAttack"
        / gen_model_name
        / f"{gen_model_name}_{target_model_name}_{dataset_name}.yaml"
    )
    if recipe_path.exists():
        recipe = load_yaml(recipe_path, dataset_cfg)
        if "gen_attack_model" in recipe and "params" in recipe["gen_attack_model"]:
            gen_params = merge_dicts(gen_params, recipe["gen_attack_model"]["params"])
        training_section = recipe.get("actual_training_params") or recipe.get("training_params") or {}
        if "gen_model_params" in training_section:
            gen_params = merge_dicts(gen_params, training_section["gen_model_params"])
        train_recipe = deepcopy(training_section)

    return gen_params, train_recipe


def rollout_student_attack(attack: TrainAttack, X: torch.Tensor, mode: str) -> torch.Tensor:
    X_adv = X
    for _ in range(attack.n_steps):
        X_adv = attack.step(X_adv, None, mode=mode)
    return X_adv


def generate_student_adv_tensor(
    attack: TrainAttack,
    X: torch.Tensor,
    batch_size: int,
    device: torch.device,
    mode: str = "val",
) -> torch.Tensor:
    X = ensure_sequence_channel_dim(X)
    batches = []
    for start in range(0, len(X), batch_size):
        X_batch = X[start : start + batch_size].to(device)
        with torch.no_grad():
            X_adv = rollout_student_attack(attack, X_batch, mode=mode)
        batches.append(X_adv.detach().cpu())
    return torch.cat(batches, dim=0)


def predict_labels(model: torch.nn.Module, X: torch.Tensor, n_classes: int) -> torch.Tensor:
    logits = model(X)
    if n_classes > 2:
        return logits.argmax(dim=-1).detach().cpu()
    return (logits >= 0.5).long().detach().cpu().view(-1)


def run_distillation_epoch(
    attack: TrainAttack,
    target_model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = torch.device("cpu"),
    alpha_l2: float = 0.0,
    n_classes: int = 2,
    mode: str = "train",
) -> Dict[str, float]:
    train_mode = optimizer is not None
    attack.gen_model.train(train_mode)
    total_loss = 0.0
    total_mse = 0.0
    total_reg = 0.0
    preds = []
    labels = []

    for X, y, teacher_adv in loader:
        X = X.to(device)
        y = y.to(device)
        teacher_adv = teacher_adv.to(device)

        if train_mode:
            optimizer.zero_grad()

        student_adv = rollout_student_attack(attack, X, mode=mode)
        mse = torch.nn.functional.mse_loss(student_adv, teacher_adv)
        reg = ((student_adv - X) ** 2).mean()
        loss = mse + alpha_l2 * reg

        if train_mode:
            loss.backward()
            optimizer.step()

        total_loss += loss.detach().item()
        total_mse += mse.detach().item()
        total_reg += reg.detach().item()
        preds.append(predict_labels(target_model, student_adv, n_classes))
        labels.append(y.detach().cpu().view(-1).long())

    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    fool_rate = float((preds != labels).mean())
    return {
        "loss": total_loss / max(1, len(loader)),
        "teacher_mse": total_mse / max(1, len(loader)),
        "perturbation_mse": total_reg / max(1, len(loader)),
        "fool_rate": fool_rate,
    }


def train_distilled_attack(
    gen_model_name: str,
    target_model_name: str,
    target_model: torch.nn.Module,
    dataset_cfg: Dict,
    train_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    valid_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Path, List[Dict[str, float]]]:
    train_X, train_y, train_teacher = train_tensors
    valid_X, valid_y, valid_teacher = valid_tensors

    save_dir = (
        ROOT
        / "checkpoints"
        / args.dataset
        / f"distilled_attack_target={target_model_name}_seed={args.seed}"
        / gen_model_name
        / args.exp_name
    )
    model_path = save_dir / f"distilled_{gen_model_name}_from_{target_model_name}_{args.seed}_{args.dataset}.pt"
    metrics_path = save_dir / f"distilled_{gen_model_name}_from_{target_model_name}_{args.seed}_{args.dataset}_metrics.csv"
    summary_path = save_dir / f"distilled_{gen_model_name}_from_{target_model_name}_{args.seed}_{args.dataset}_summary.yaml"

    gen_params, recipe = load_gen_recipe(gen_model_name, target_model_name, args.dataset, dataset_cfg)
    gen_model = get_model(gen_model_name, gen_params, device=str(device))

    req_grad(target_model, False)
    target_model.eval()

    student_train_eps = args.student_train_eps if args.student_train_eps is not None else args.teacher_eps
    student_eps = args.student_eps if args.student_eps is not None else args.teacher_eps
    student_steps = args.student_steps if args.student_steps is not None else args.teacher_steps

    attack = TrainAttack(
        model=target_model,
        gen_model=gen_model,
        criterion=get_classifier_criterion(dataset_cfg["num_classes"]),
        estimator=None,
        train_eps=student_train_eps,
        eps=student_eps,
        n_steps=student_steps,
        n_classes=dataset_cfg["num_classes"],
    )

    if model_path.exists() and not args.force_distill:
        state_dict = torch.load(model_path, map_location=device)
        attack.gen_model.load_state_dict(state_dict)
        train_adv = generate_student_adv_tensor(attack, train_X, args.batch_size, device, mode="val")
        valid_adv = generate_student_adv_tensor(attack, valid_X, args.batch_size, device, mode="val")
        train_metrics = evaluate_attack_outputs(
            target_model, train_X, train_adv, train_y, dataset_cfg["num_classes"]
        )
        valid_metrics = evaluate_attack_outputs(
            target_model, valid_X, valid_adv, valid_y, dataset_cfg["num_classes"]
        )
        summary_rows = [
            {
                "attack_kind": "distilled",
                "target_model": target_model_name,
                "generator_model": gen_model_name,
                "split": "train",
                "clean_acc": train_metrics["clean_acc"],
                "acc": train_metrics["acc"],
                "fooling_rate": train_metrics["fooling_rate"],
                "eff": train_metrics["eff"],
                "f1": train_metrics["f1"],
                "teacher_mse": float(torch.nn.functional.mse_loss(train_adv, train_teacher).item()),
                "checkpoint_path": str(model_path),
            },
            {
                "attack_kind": "distilled",
                "target_model": target_model_name,
                "generator_model": gen_model_name,
                "split": "valid",
                "clean_acc": valid_metrics["clean_acc"],
                "acc": valid_metrics["acc"],
                "fooling_rate": valid_metrics["fooling_rate"],
                "eff": valid_metrics["eff"],
                "f1": valid_metrics["f1"],
                "teacher_mse": float(torch.nn.functional.mse_loss(valid_adv, valid_teacher).item()),
                "checkpoint_path": str(model_path),
            },
        ]
        return model_path, summary_rows

    optimizer_name = recipe.get("optimizer_name", "Adam")
    lr = recipe.get("optimizer_params", {}).get("lr", args.distill_lr)
    optimizer_cls = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_cls(attack.gen_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=recipe.get("scheduler_params", {}).get("step_size", args.distill_step_size),
        gamma=recipe.get("scheduler_params", {}).get("gamma", args.distill_gamma),
    )
    alpha_l2 = recipe.get("alpha_l2", args.distill_alpha_l2)
    max_epochs = recipe.get("n_epochs", args.distill_epochs)

    train_loader = DataLoader(
        DistillationDataset(train_X, train_y, train_teacher),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        DistillationDataset(valid_X, valid_y, valid_teacher),
        batch_size=args.batch_size,
        shuffle=False,
    )

    history = []
    best_state = deepcopy(attack.gen_model.state_dict())
    best_val_loss = math.inf
    patience = 0

    for epoch in range(1, max_epochs + 1):
        train_metrics = run_distillation_epoch(
            attack=attack,
            target_model=target_model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha_l2=alpha_l2,
            n_classes=dataset_cfg["num_classes"],
            mode="train",
        )
        valid_metrics = run_distillation_epoch(
            attack=attack,
            target_model=target_model,
            loader=valid_loader,
            optimizer=None,
            device=device,
            alpha_l2=alpha_l2,
            n_classes=dataset_cfg["num_classes"],
            mode="val",
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_teacher_mse": train_metrics["teacher_mse"],
            "train_perturbation_mse": train_metrics["perturbation_mse"],
            "train_fool_rate": train_metrics["fool_rate"],
            "valid_loss": valid_metrics["loss"],
            "valid_teacher_mse": valid_metrics["teacher_mse"],
            "valid_perturbation_mse": valid_metrics["perturbation_mse"],
            "valid_fool_rate": valid_metrics["fool_rate"],
        }
        history.append(row)
        print(
            f"[distill:{target_model_name}->{gen_model_name}] epoch={epoch} "
            f"train_mse={train_metrics['teacher_mse']:.6f} valid_mse={valid_metrics['teacher_mse']:.6f} "
            f"valid_fool={valid_metrics['fool_rate']:.4f}"
        )

        if valid_metrics["loss"] < best_val_loss:
            best_val_loss = valid_metrics["loss"]
            best_state = deepcopy(attack.gen_model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.distill_patience:
                break

    ensure_dir(save_dir)
    attack.gen_model.load_state_dict(best_state)
    torch.save(attack.gen_model.state_dict(), model_path)
    save_csv(metrics_path, history)
    with summary_path.open("w") as handle:
        yaml.safe_dump(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "target_model": target_model_name,
                "generator_model": gen_model_name,
                "teacher_eps": args.teacher_eps,
                "teacher_steps": args.teacher_steps,
                "student_train_eps": student_train_eps,
                "student_eps": student_eps,
                "student_steps": student_steps,
                "generator_params": gen_params,
                "best_valid_loss": float(best_val_loss),
                "checkpoint_path": str(model_path),
            },
            handle,
            sort_keys=False,
        )

    train_adv = generate_student_adv_tensor(attack, train_X, args.batch_size, device, mode="val")
    valid_adv = generate_student_adv_tensor(attack, valid_X, args.batch_size, device, mode="val")
    train_metrics = evaluate_attack_outputs(
        target_model, train_X, train_adv, train_y, dataset_cfg["num_classes"]
    )
    valid_metrics = evaluate_attack_outputs(
        target_model, valid_X, valid_adv, valid_y, dataset_cfg["num_classes"]
    )
    summary_rows = [
        {
            "attack_kind": "distilled",
            "target_model": target_model_name,
            "generator_model": gen_model_name,
            "split": "train",
            "clean_acc": train_metrics["clean_acc"],
            "acc": train_metrics["acc"],
            "fooling_rate": train_metrics["fooling_rate"],
            "eff": train_metrics["eff"],
            "f1": train_metrics["f1"],
            "teacher_mse": float(torch.nn.functional.mse_loss(train_adv, train_teacher).item()),
            "checkpoint_path": str(model_path),
        },
        {
            "attack_kind": "distilled",
            "target_model": target_model_name,
            "generator_model": gen_model_name,
            "split": "valid",
            "clean_acc": valid_metrics["clean_acc"],
            "acc": valid_metrics["acc"],
            "fooling_rate": valid_metrics["fooling_rate"],
            "eff": valid_metrics["eff"],
            "f1": valid_metrics["f1"],
            "teacher_mse": float(torch.nn.functional.mse_loss(valid_adv, valid_teacher).item()),
            "checkpoint_path": str(model_path),
        },
    ]
    return model_path, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-run distilled attack launcher, similar in spirit to attack_run.py."
    )
    parser.add_argument("--dataset", default="PowerCons")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--exp-name", default="distill_attack_run")
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--gen-model", required=True)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-step-size", type=int, default=8)
    parser.add_argument("--classifier-gamma", type=float, default=0.9)
    parser.add_argument("--classifier-patience", type=int, default=60)
    parser.add_argument("--teacher-eps", type=float, default=0.3)
    parser.add_argument("--teacher-steps", type=int, default=10)
    parser.add_argument("--distill-epochs", type=int, default=30)
    parser.add_argument("--distill-lr", type=float, default=1e-3)
    parser.add_argument("--distill-step-size", type=int, default=10)
    parser.add_argument("--distill-gamma", type=float, default=0.9)
    parser.add_argument("--distill-patience", type=int, default=20)
    parser.add_argument("--distill-alpha-l2", type=float, default=1e-4)
    parser.add_argument("--student-train-eps", type=float, default=None)
    parser.add_argument("--student-eps", type=float, default=None)
    parser.add_argument("--student-steps", type=int, default=None)
    parser.add_argument("--force-classifier", action="store_true")
    parser.add_argument("--force-teacher", action="store_true")
    parser.add_argument("--force-distill", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    if not dataset_exists(args.dataset):
        raise FileNotFoundError(
            f"Dataset {args.dataset} not found in {ROOT / 'data' / 'UCR'} or {ROOT / 'data' / 'UEA'}."
        )

    args.force_classifiers = args.force_classifier
    args.force_teachers = args.force_teacher
    args.force_distill = args.force_distill

    fix_seed(args.seed)
    dataset_cfg = load_dataset_cfg(args.dataset)

    X_train, y_train, X_valid, y_valid = load_data(args.dataset)
    X_train, X_valid, y_train, y_valid = transform_data(
        X_train,
        X_valid,
        y_train,
        y_valid,
        slice_data=False,
    )

    classifier_train_loader = DataLoader(
        MyDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    classifier_valid_loader = DataLoader(
        MyDataset(X_valid, y_valid),
        batch_size=args.batch_size,
        shuffle=False,
    )
    teacher_train_loader = DataLoader(
        MyDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=False,
    )
    teacher_valid_loader = DataLoader(
        MyDataset(X_valid, y_valid),
        batch_size=args.batch_size,
        shuffle=False,
    )

    classifier, classifier_path, classifier_metrics = train_classifier(
        model_name=args.target_model,
        dataset_cfg=dataset_cfg,
        train_loader=classifier_train_loader,
        valid_loader=classifier_valid_loader,
        args=args,
        device=device,
    )
    print(
        f"[classifier:{args.target_model}] ready path={classifier_path} "
        f"valid_acc={classifier_metrics['accuracy']:.4f} valid_f1={classifier_metrics['f1']:.4f}"
    )

    train_teacher, train_teacher_path = generate_or_load_teacher(
        model=classifier,
        target_model_name=args.target_model,
        split_name="train",
        loader=teacher_train_loader,
        args=args,
        dataset_cfg=dataset_cfg,
        device=device,
    )
    valid_teacher, valid_teacher_path = generate_or_load_teacher(
        model=classifier,
        target_model_name=args.target_model,
        split_name="valid",
        loader=teacher_valid_loader,
        args=args,
        dataset_cfg=dataset_cfg,
        device=device,
    )

    teacher_train_metrics = evaluate_attack_outputs(
        classifier, X_train, train_teacher, y_train, dataset_cfg["num_classes"]
    )
    teacher_valid_metrics = evaluate_attack_outputs(
        classifier, X_valid, valid_teacher, y_valid, dataset_cfg["num_classes"]
    )
    print(
        f"[ifgsm:{args.target_model}] split=train clean_acc={teacher_train_metrics['clean_acc']:.4f} "
        f"acc={teacher_train_metrics['acc']:.4f} fooling_rate={teacher_train_metrics['fooling_rate']:.4f} "
        f"eff={teacher_train_metrics['eff']:.4f}"
    )
    print(
        f"[ifgsm:{args.target_model}] split=valid clean_acc={teacher_valid_metrics['clean_acc']:.4f} "
        f"acc={teacher_valid_metrics['acc']:.4f} fooling_rate={teacher_valid_metrics['fooling_rate']:.4f} "
        f"eff={teacher_valid_metrics['eff']:.4f}"
    )
    print(f"[teacher:{args.target_model}] train={train_teacher_path} valid={valid_teacher_path}")

    teacher_rows = [
        {
            "attack_kind": "ifgsm_teacher",
            "target_model": args.target_model,
            "generator_model": "ifgsm",
            "split": "train",
            "clean_acc": teacher_train_metrics["clean_acc"],
            "acc": teacher_train_metrics["acc"],
            "fooling_rate": teacher_train_metrics["fooling_rate"],
            "eff": teacher_train_metrics["eff"],
            "f1": teacher_train_metrics["f1"],
            "teacher_mse": 0.0,
            "checkpoint_path": str(train_teacher_path),
        },
        {
            "attack_kind": "ifgsm_teacher",
            "target_model": args.target_model,
            "generator_model": "ifgsm",
            "split": "valid",
            "clean_acc": teacher_valid_metrics["clean_acc"],
            "acc": teacher_valid_metrics["acc"],
            "fooling_rate": teacher_valid_metrics["fooling_rate"],
            "eff": teacher_valid_metrics["eff"],
            "f1": teacher_valid_metrics["f1"],
            "teacher_mse": 0.0,
            "checkpoint_path": str(valid_teacher_path),
        },
    ]

    model_path, summary_rows = train_distilled_attack(
        gen_model_name=args.gen_model,
        target_model_name=args.target_model,
        target_model=classifier,
        dataset_cfg=dataset_cfg,
        train_tensors=(X_train, y_train, train_teacher),
        valid_tensors=(X_valid, y_valid, valid_teacher),
        args=args,
        device=device,
    )

    for row in summary_rows:
        print(
            f"[distill:{args.target_model}->{args.gen_model}] split={row['split']} "
            f"clean_acc={row['clean_acc']:.4f} acc={row['acc']:.4f} "
            f"fooling_rate={row['fooling_rate']:.4f} eff={row['eff']:.4f} "
            f"teacher_mse={row['teacher_mse']:.6f}"
        )
    print(f"[distill:{args.target_model}->{args.gen_model}] saved={model_path}")

    all_rows = teacher_rows + summary_rows
    summary_df = pd.DataFrame(all_rows)
    if not summary_df.empty:
        print("\nFinal distilled attack metrics:")
        print(summary_df.round(4).to_string(index=False))

    report_dir = ROOT / "artifacts" / "summary" / args.dataset / args.exp_name
    report_dir.mkdir(parents=True, exist_ok=True)
    save_csv(
        report_dir / f"distill_attack_run_{args.target_model}_{args.gen_model}.csv",
        all_rows,
    )
    print(
        f"[summary] {report_dir / f'distill_attack_run_{args.target_model}_{args.gen_model}.csv'}"
    )


if __name__ == "__main__":
    main()