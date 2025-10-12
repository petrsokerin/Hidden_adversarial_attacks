from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_attacks import BaseIterativeAttack
from .procedures import BatchIterativeAttack
from src.estimation import BaseEstimator


class _STClamp(torch.autograd.Function):
    """Straight-through estimator for clamping operations in BPDA."""
    @staticmethod
    def forward(ctx, x, lo, hi):
        return x.clamp(lo, hi)

    @staticmethod
    def backward(ctx, g):
        return g, None, None


def st_clamp(x, lo, hi, use_st: bool):
    """Clamp with optional straight-through estimator."""
    return _STClamp.apply(x, lo, hi) if use_st else x.clamp(lo, hi)


class ModelBasedAttack(BaseIterativeAttack, BatchIterativeAttack):
    """
    Base class for Model-Based Attacks (MBA) using attacker models.
    
    This attack uses a pre-trained attacker model (generator) to generate adversarial perturbations
    without requiring access to the victim model's gradients.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        attacker_model: torch.nn.Module,
        criterion: torch.nn.Module,
        estimator: BaseEstimator,
        eps: float = 0.5,
        n_steps: int = 1,
        n_classes: int = 2,
        clamp: Optional[Tuple[float, float]] = None,
        logger=None,
        *args,
        **kwargs
    ) -> None:
        BaseIterativeAttack.__init__(self, model=model, n_steps=n_steps, n_classes=n_classes)
        BatchIterativeAttack.__init__(self, estimator=estimator, logger=logger, n_classes=n_classes)
        
        self.attacker_model = attacker_model
        self.criterion = criterion
        self.eps = eps
        self.clamp = clamp
        
        # Set surrogate to eval mode
        self.attacker_model.eval()
        for param in self.attacker_model.parameters():
            param.requires_grad_(False)

    def get_loss(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Get loss for logging purposes (MBA attacks don't use victim model gradients).
        
        Args:
            X: Input tensor
            y_true: True labels
            
        Returns:
            Loss value for logging
        """
        with torch.no_grad():
            y_pred = self.model(X)
            
            if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                y_true = y_true.view(-1).long()
            
            loss = self.criterion(y_pred, y_true)
            return loss

    def get_adv_data(
        self,
        X: torch.Tensor,
        y_true: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate adversarial examples using the surrogate model.
        
        Args:
            X: Input tensor
            y_true: True labels (not used in MBA, kept for compatibility)
            
        Returns:
            Adversarial examples
        """
        with torch.no_grad():
            delta = self.eps * torch.tanh(self.attacker_model(X))
            X_adv = X + delta
            
            if self.clamp is not None:
                X_adv = torch.clamp(X_adv, self.clamp[0], self.clamp[1])
                
        return X_adv

    def step(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Single step of the MBA attack.
        
        Args:
            X: Input tensor
            y_true: True labels
            
        Returns:
            Adversarial examples
        """
        return self.get_adv_data(X, y_true)


class IterativeModelBasedAttack(ModelBasedAttack):
    """
    Iterative Model-Based Attack with advanced features.
    
    Supports momentum, projections, BPDA, and other advanced techniques.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        attacker_model: torch.nn.Module,
        criterion: torch.nn.Module,
        estimator: BaseEstimator,
        eps: float = 0.5,
        n_steps: int = 10,
        alpha: Optional[float] = None,
        n_classes: int = 2,
        clamp: Optional[Tuple[float, float]] = None,
        rand_init: bool = True,
        use_sign: bool = False,
        equal_eps: bool = False,
        bpda: bool = True,
        proj: str = "none",  # "none" | "linf" | "l2"
        proj_equal_eps: bool = False,
        momentum_mu: float = 0.0,  # MI-FGSM momentum
        step_normalize: Optional[str] = None,  # "meanabs" | "l2" | "linf" | None
        step_noise_std: float = 0.0,
        logger=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            model=model,
            attacker_model=attacker_model,
            criterion=criterion,
            estimator=estimator,
            eps=eps,
            n_steps=n_steps,
            n_classes=n_classes,
            clamp=clamp,
            logger=logger,
            *args,
            **kwargs
        )
        
        self.alpha = alpha or (1.0 * eps / max(1, n_steps))
        self.rand_init = rand_init
        self.use_sign = use_sign
        self.equal_eps = equal_eps
        self.bpda = bpda
        self.proj = proj
        self.proj_equal_eps = proj_equal_eps
        self.momentum_mu = momentum_mu
        self.step_normalize = step_normalize
        self.step_noise_std = step_noise_std

    @staticmethod
    def _project(delta: torch.Tensor, eps: float, mode: str, equal_eps: bool, bpda: bool) -> torch.Tensor:
        """Project perturbation to epsilon ball."""
        if mode == "none":
            return delta
        if mode == "linf":
            delta = st_clamp(delta, -eps, eps, use_st=bpda)
            if equal_eps:
                amax = delta.detach().abs().flatten(1).amax(dim=1)  # (B,)
                scale = (eps / (amax + 1e-12)).view(-1, *[1] * (delta.dim() - 1))
                delta = delta * scale
                delta = st_clamp(delta, -eps, eps, use_st=bpda)
            return delta
        if mode == "l2":
            flat = delta.flatten(1)
            nrm = torch.norm(flat, p=2, dim=1).view(-1, *[1] * (delta.dim() - 1)) + 1e-12
            # Project to L2 ball
            delta = delta * torch.clamp(eps / nrm, max=1.0)
            if equal_eps:
                delta = delta * (eps / (nrm + 1e-12))
            return delta
        raise ValueError(f"Unknown projection mode: {mode}")

    @staticmethod
    def _normalize_step(d: torch.Tensor, kind: Optional[str] = None) -> torch.Tensor:
        """Normalize step direction."""
        if kind in (None, "none"):
            return d
        if kind == "meanabs":
            s = d.abs().mean(dim=tuple(range(1, d.dim())), keepdim=True) + 1e-12
            return d / s
        if kind == "l2":
            s = torch.norm(d.flatten(1), p=2, dim=1).view(-1, *[1] * (d.dim() - 1)) + 1e-12
            return d / s
        if kind == "linf":
            s = d.abs().flatten(1).amax(dim=1).view(-1, *[1] * (d.dim() - 1)) + 1e-12
            return d / s
        raise ValueError(f"Unknown normalize kind: {kind}")

    def step(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Single step of the iterative MBA attack.
        
        Args:
            X: Input tensor
            y_true: True labels
            
        Returns:
            Adversarial examples
        """
        x0 = X.clone()
        
        # Random initialization
        if self.rand_init:
            if self.proj == "linf":
                delta0 = torch.empty_like(x0).uniform_(-self.eps, self.eps)
            elif self.proj == "l2":
                delta0 = torch.randn_like(x0)
                delta0 = self._project(delta0, self.eps, "l2", False, self.bpda)
            else:  # none
                delta0 = torch.empty_like(x0).uniform_(-self.eps, self.eps)
            x_adv = x0 + delta0
        else:
            x_adv = x0.clone()

        g = torch.zeros_like(x_adv)  # momentum buffer

        for _ in range(self.n_steps):
            # Get step direction from surrogate model
            if self.bpda:
                # Use BPDA: replace non-differentiable operations with differentiable approximations
                step_dir = self.attacker_model(x_adv)
            else:
                with torch.no_grad():
                    step_dir = self.attacker_model(x_adv)
            
            # Apply activation
            step_dir = step_dir.sign() if self.use_sign else torch.tanh(step_dir)
            
            # Normalize step
            if self.step_normalize:
                step_dir = self._normalize_step(step_dir, self.step_normalize)
            
            # Add noise
            if self.step_noise_std > 0:
                step_dir = step_dir + self.step_noise_std * torch.randn_like(step_dir)

            # Momentum (MI-FGSM)
            if self.momentum_mu > 0.0:
                g = self.momentum_mu * g + step_dir
                step_dir = g.sign() if self.use_sign else g

            # Update adversarial example
            x_adv = x_adv + self.alpha * step_dir

            # Project to epsilon ball
            delta = x_adv - x0
            delta = self._project(delta, self.eps, self.proj, self.proj_equal_eps, self.bpda)
            x_adv = x0 + delta

            # Apply data clamping
            if self.clamp is not None:
                x_adv = st_clamp(x_adv, self.clamp[0], self.clamp[1], use_st=self.bpda)

        return x_adv
