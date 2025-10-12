from abc import abstractmethod

import torch


class BaseModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.self_supervised = False


class BaseSelfSupervisedModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.self_supervised = True

    @abstractmethod
    def train_embedding(self, *args, **kwargs) -> None:
        pass
