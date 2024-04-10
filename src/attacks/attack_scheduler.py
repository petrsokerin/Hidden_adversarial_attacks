from abc import abstractmethod

from src.attacks import BaseIterativeAttack


class AttackScheduler:
    def __init__(self, attack: BaseIterativeAttack):
        self.attack = attack

    @abstractmethod
    def step(self):
        pass


class StepAttackScheduler(AttackScheduler):
    def __init__(
        self,
        attack: BaseIterativeAttack,
        attack_step_size: int = 1,
        attack_gamma: float = 1.0,
    ) -> None:
        super().__init__(attack)

        if attack_gamma < 0 or attack_gamma > 1:
            raise ValueError("Attack gamma should be between 0 and 1")

        self.attack_step_size = attack_step_size
        self.attack_gamma = attack_gamma
        self.iters = 0

    def step(self) -> BaseIterativeAttack:
        self.iters += 1
        if self.iters == self.attack_step_size:
            self.attack.eps *= self.attack_gamma
            self.iters = 0
        return self.attack
