from base_attacks import IterativeAttack
from regularizers import reg_neigh, reg_disc

class FGSMAttack(IterativeAttack):
    def __init__(self, model, criterion, eps, device='cpu'):
        super().__init__(model, device=device)
        self.criterion = criterion
        self.eps = eps

    def get_loss(self, x, y_true):
        x.requires_grad = True
        self.model.zero_grad()
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_true)
        return loss
    
    def apply_attack(self, x, loss):
        loss.backward()
        grad_sign = x.grad.sign()
        x_adv = x.data + self.eps * grad_sign
        return x_adv

    def step(self, x, y_true):
        loss = self.get_loss(x, y_true)
        x_adv = self.apply_attack(x, loss)
        return x_adv
    

class FGSMRegNeighAttack(FGSMAttack):
    def __init__(self, model, criterion, eps, alpha=0.0, device='cpu'):
        super().__init__(model, criterion, eps, device=device)
        self.alpha = alpha

    def step(self, x, y_true):
        loss = self.attack_run(x, y_true)

        reg_value = reg_neigh(x, self.alpha)
        loss = loss - reg_value

        x_adv = self.apply_attack(x, loss)
        return x_adv


class FGSMRegDiscAttack(FGSMAttack):
    def __init__(self, model, criterion, eps, disc_models, alpha, device='cpu', use_sigmoid=False):
        super().__init__(model, criterion, eps, device=device)
        self.alpha = alpha
        self.disc_models = disc_models
        self.use_sigmoid = use_sigmoid

    def forward(self, x, y_true):
        reg_value = reg_disc(x, self.alpha)
        loss = loss - reg_value

        x_adv = self.apply_attack(x, loss)
        return x_adv

