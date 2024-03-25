from abc import ABC, abstractmethod

class BaseAttack(ABC):
    def __init__(self, model, n_steps=50):
        self.model = model
        self.device= next(model.parameters()).device
        self.n_steps = n_steps

    @abstractmethod
    def step(self, X, y_true):
        raise NotImplementedError("This method should be implemented by subclasses.")
