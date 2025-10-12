import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader

from .train import Trainer, EarlyStopper


class MaxEarlyStopper:
    """Early stopper for maximizing metrics (like victim loss in MBA attacks)."""
    
    def __init__(
        self,
        patience: int = 1,
        min_delta: float = 0.0,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_loss = -np.inf

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss > self.max_validation_loss:
            self.max_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss < (self.max_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class AttackerTrainer(Trainer):
    """
    Trainer for attacker models (surrogate models) that generate adversarial perturbations.
    
    This trainer maximizes victim model loss while keeping perturbations small.
    """
    
    def __init__(
        self,
        attacker_model: torch.nn.Module,
        victim_model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        n_epochs: int = 30,
        early_stop_patience: int = None,
        logger: Any = None,
        print_every: int = 5,
        device: str = "cpu",
        eps: float = 0.5,
        alpha_l2: float = 1e-3,
        is_clamped: bool = False,
        is_debugged: bool = False,
    ) -> None:
        # Initialize base Trainer with attacker_model as the main model
        super().__init__(
            model=attacker_model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=n_epochs,
            early_stop_patience=early_stop_patience,
            logger=logger,
            print_every=print_every,
            device=device,
        )
        
        self.attacker_model = attacker_model
        self.victim_model = victim_model
        self.eps = eps
        self.alpha_l2 = alpha_l2
        self.is_clamped = is_clamped
        self.is_debugged = is_debugged
        
        # Freeze victim model parameters
        self.victim_model.eval()
        for param in self.victim_model.parameters():
            param.requires_grad_(False)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train attacker model for one epoch."""
        self.attacker_model.train()
        
        epoch_victim_loss = 0.0
        epoch_acc = 0.0
        epoch_reg = 0.0
        n_samples = 0
        
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            # Generate perturbation using attacker model
            delta = self.eps * torch.tanh(self.attacker_model(x))
            
            if self.is_clamped:
                delta = torch.clamp(delta, -1, 1)
            
            # Create adversarial example
            x_adv = x + delta
            
            # Get victim model predictions
            with torch.no_grad():
                logits = self.victim_model(x_adv)
                victim_loss = self.criterion(logits, y)
                acc = (logits.argmax(1) == y).float().mean().item()
            
            # L2 regularization
            reg = self.alpha_l2 * (delta**2).mean()
            
            # Attacker loss: maximize victim loss (minimize negative victim loss)
            loss = -(victim_loss - reg)
            
            # Debug information
            if self.is_debugged:
                print(f'\nx={((x[0].detach().cpu()**2).mean())**0.5}')
                print(f'delta={((delta[0].detach().cpu()**2).mean())**0.5}')
                print(f'y={y.detach().cpu()[0]}|logits={logits.detach().cpu().argmax(1)[0]}')
                print(f'vloss = {victim_loss}| reg = {reg}')
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            epoch_victim_loss += victim_loss.item() * x.size(0)
            epoch_acc += acc * x.size(0)
            epoch_reg += reg.item() * x.size(0)
            n_samples += x.size(0)
        
        # Average metrics
        avg_victim_loss = epoch_victim_loss / n_samples
        avg_acc = epoch_acc / n_samples
        avg_reg = epoch_reg / n_samples
        
        return {
            'victim_loss': avg_victim_loss,
            'accuracy': avg_acc,
            'regularization': avg_reg,
            'attacker_loss': -(avg_victim_loss - avg_reg)
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate attacker model for one epoch."""
        self.attacker_model.eval()
        
        val_victim_loss = 0.0
        val_acc = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Generate perturbation
                delta = self.eps * torch.tanh(self.attacker_model(x))
                
                if self.is_clamped:
                    delta = torch.clamp(delta, -1, 1)
                
                # Create adversarial example
                x_adv = x + delta
                
                # Get victim model predictions
                logits = self.victim_model(x_adv)
                victim_loss = self.criterion(logits, y)
                acc = (logits.argmax(1) == y).float().mean().item()
                
                val_victim_loss += victim_loss.item() * x.size(0)
                val_acc += acc * x.size(0)
                n_samples += x.size(0)
        
        return {
            'victim_loss': val_victim_loss / n_samples,
            'accuracy': val_acc / n_samples
        }
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Train the attacker model."""
        # Use MaxEarlyStopper for maximizing victim loss
        stopper = MaxEarlyStopper(patience=self.early_stop_patience) if val_loader else None
        
        best_val_loss = -np.inf  # We maximize victim loss
        best_state_dict = self.attacker_model.state_dict()
        history = {'train': [], 'val': []}
        
        for epoch in range(1, self.n_epochs + 1):
            # Training
            train_metrics = self.train_epoch(train_loader)
            history['train'].append(train_metrics)
            
            # Validation
            if val_loader:
                val_metrics = self.validate_epoch(val_loader)
                history['val'].append(val_metrics)
                
                # Save best model (highest victim loss)
                if val_metrics['victim_loss'] > best_val_loss:
                    best_val_loss = val_metrics['victim_loss']
                    best_state_dict = self.attacker_model.state_dict()
                
                # Early stopping
                if stopper and stopper.early_stop(val_metrics['victim_loss']):
                    if self.print_every:
                        print(f'⏹ Early stopping at epoch {epoch}')
                    break
            else:
                val_metrics = train_metrics
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            if self.logger:
                for key, value in train_metrics.items():
                    self.logger.add_scalar(f'train/{key}', value, epoch)
                if val_loader:
                    for key, value in val_metrics.items():
                        self.logger.add_scalar(f'val/{key}', value, epoch)
            
            # Print progress
            if self.print_every and epoch % self.print_every == 0:
                msg = f'Epoch {epoch:02d} | train-victim-loss {train_metrics["victim_loss"]:.4f} | train-acc {train_metrics["accuracy"]:.4f}'
                if val_loader:
                    msg += f' | val-victim-loss {val_metrics["victim_loss"]:.4f} | val-acc {val_metrics["accuracy"]:.4f}'
                print(msg)
        
        # Load best model
        if val_loader:
            self.attacker_model.load_state_dict(best_state_dict)
        
        return {
            'history': history,
            'best_val_loss': best_val_loss,
            'best_state_dict': best_state_dict
        }