from abc import ABC

from abc import ABC, abstractmethod

class BaseAttack(ABC):
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)

    @abstractmethod
    def step(self, x, y_true):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @abstractmethod
    def forward(self, x, y_true):
        # The forward method might not be necessary if it just calls step.
        # This implementation assumes that step and forward might have different behaviors in subclasses.
        return self.step(x.to(self.device), y_true.to(self.device))


class IterativeAttack(BaseAttack):
    def __init__(self, parameter):
        super().__init__()
        self.parameter = parameter

    def step(self, x, y_true):
        # Default step behavior for iterative attacks. This might be overridden.
        # Assuming step logic for iterative attacks is defined here.
        super().step(x, y_true)

    def forward(self, x, y_true):
        x_adv = x.clone().detach().to(self.device)
        for _ in range(self.n_steps):
            x_adv = self.step(x_adv, y_true)
        return x_adv