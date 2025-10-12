import os
import warnings
import time
import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from clearml import Task

from src.config import get_criterion, get_model, get_optimizer, get_scheduler
from src.data import MyDataset, load_data, transform_data
from src.training.train_attacker import AttackerTrainer
from src.utils import fix_seed, save_config, save_compiled_config

warnings.filterwarnings("ignore")

CONFIG_NAME = "train_attacker_config"
CONFIG_PATH = "config"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    start_time = time.time()
    
    # Initialize ClearML if enabled
    if cfg.get("log_clearml", False):
        if cfg.get("author", "") == "":
            raise ValueError("You need to set your name in config when using ClearML")
        
        task_name = f"surrogate_{cfg['attacker_model']['name']}_{cfg['model_id']}_{cfg['dataset']['name']}"
        task = Task.init(
            project_name=cfg.get("project_name", "surrogate_training"),
            task_name=task_name,
            auto_connect_frameworks={"pytorch": False}
        )
        task.connect(cfg)
        logger = task.get_logger()
    else:
        logger = None
    
    # Set random seed
    fix_seed(cfg['model_id'])
    
    # Load data
    print(f"Dataset: {cfg['dataset']['name']}")
    X_train, y_train, X_test, y_test = load_data(cfg["dataset"]["name"])
    X_train, X_test, y_train, y_test = transform_data(
        X_train, X_test, y_train, y_test, slice_data=cfg.get("slice", False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        MyDataset(X_train, y_train), 
        batch_size=cfg["batch_size"], 
        shuffle=True
    )
    test_loader = DataLoader(
        MyDataset(X_test, y_test), 
        batch_size=cfg["batch_size"], 
        shuffle=False
    )
    
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load victim model
    victim_model_path = os.path.join(
        cfg["model_folder"],
        f"model_{cfg['victim_model']['name']}_{cfg['model_id']}_{cfg['dataset']['name']}.pt"
    )
    
    if not os.path.exists(victim_model_path):
        raise FileNotFoundError(f"Victim model not found at {victim_model_path}")
    
    victim_model = get_model(
        cfg["victim_model"]["name"],
        cfg["victim_model"]["params"],
        path=victim_model_path,
        device=device,
        train_mode=cfg["victim_model"]["attack_train_mode"],
    )
    
    print(f"Loaded victim model: {cfg['victim_model']['name']}")
    
    # Create surrogate model
    attacker_model = get_model(
        cfg["attacker_model"]["name"],
        cfg["attacker_model"]["params"],
        device=device,
        train_mode=True,
    )
    
    print(f"Created surrogate model: {cfg['attacker_model']['name']}")
    
    # Create criterion
    criterion = get_criterion(cfg["criterion_name"], cfg["criterion_params"])
    
    # Create optimizer
    optimizer = get_optimizer(
        cfg["optimizer"]["name"], 
        attacker_model.parameters(), 
        cfg["optimizer"]["params"]
    )
    
    # Create scheduler
    scheduler = get_scheduler(
        cfg["scheduler"]["name"], 
        optimizer, 
        cfg["scheduler"]["params"]
    )
    
    # Create save directory
    os.makedirs(cfg["attacker_save_path"], exist_ok=True)
    
    # Generate save path
    surrogate_name = f"surrogate_{cfg['attacker_model']['name']}_{cfg['model_id']}_{cfg['dataset']['name']}"
    save_path = os.path.join(cfg["attacker_save_path"], f"{surrogate_name}.pth")
    
    # Create AttackerTrainer
    trainer = AttackerTrainer(
        attacker_model=attacker_model,
        victim_model=victim_model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=cfg["epochs"],
        early_stop_patience=cfg["patience"],
        logger=logger,
        print_every=cfg["verbose_every"],
        device=device,
        eps=cfg["eps"],
        alpha_l2=cfg["alpha_l2"],
        is_clamped=cfg.get("is_clamped", False),
        is_debugged=cfg.get("is_debugged", False),
    )
    
    # Train attacker model
    print("Starting attacker model training...")
    
    results = trainer.train(train_loader, test_loader)
    
    # Save model
    torch.save(attacker_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Extract final metrics
    val_loss = results['best_val_loss']
    val_acc = results['history']['val'][-1]['accuracy'] if results['history']['val'] else 0.0
    
    print(f"Training completed!")
    print(f"Final validation loss: {val_loss:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    
    # Save configuration
    if not cfg.get("test_run", False):
        save_config(cfg["save_path"], CONFIG_PATH, CONFIG_NAME, surrogate_name)
        save_compiled_config(cfg, cfg["save_path"], surrogate_name)
    
    # Log final metrics
    if logger:
        logger.report_scalar("final", "val_loss", val_loss)
        logger.report_scalar("final", "val_acc", val_acc)
    
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
