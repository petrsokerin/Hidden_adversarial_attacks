from typing import List
import torch
import torch.nn.functional as F

from .utils import req_grad


def simba_binary(model, loss_func, x, y_true, eps):
    rand_ind = torch.randint(x.shape[1], (1,))[0]
    mask = torch.zeros(x.shape)
    mask[:, rand_ind] = mask[:, rand_ind] + 1
    mask = mask.to(x.device)

    y_pred = model(x)

    p_orig = torch.max(y_pred, 1-y_pred)

    x_minus = x - mask * eps
    y_pred_minus = model(x_minus)
    p_minus = torch.max(y_pred_minus, 1-y_pred_minus)
    
    x_plus = x + mask * eps
    y_pred_plus = model(x_plus)
    p_plus = torch.max(y_pred_plus, 1-y_pred_plus)

    x_all = torch.cat([x, x_minus, x_plus]).view(3, *x.shape)

    mask = torch.argmin(torch.hstack([p_orig, p_minus, p_plus]), dim=1)
    mask_reshaped = mask.view(-1, 1, 1).repeat(3, 1, x_all.shape[2]).view(x_all.shape)

    x_adv = x_all.gather(0, mask_reshaped)[0]
    return x_adv


def simba_binary_reg(model, loss_func, x, y_true, eps, alpha):
    x_adv = simba_binary(model, loss_func, x, y_true, eps)
    reg_value = reg_neigh(x_adv, alpha)

    loss = - reg_value
    grad_ = torch.autograd.grad(loss, x, retain_graph=True)[0]
    x_adv = x_adv.data + grad_

    return x_adv


def simba_binary_disc_reg(
    model, 
    loss_func, 
    x, 
    y_true, 
    eps, 
    alpha, 
    disc_models
):
    x_adv = simba_binary(model, loss_func, x, y_true, eps)
    reg_value = reg_disc(x_adv, alpha, disc_models)

    loss = - reg_value
    grad_ = torch.autograd.grad(loss, x, retain_graph=True)[0]
    x_adv = x_adv.data + grad_

    return x_adv


def fgsm_attack(model, loss_func, x, y_true, eps):
    y_pred = model(x)
    loss_val = loss_func(y_pred, y_true)

    grad_ = torch.autograd.grad(loss_val, x, retain_graph=True)[0]
    x_adv = x.data + eps * torch.sign(grad_)

    return x_adv


def deepfool_attack(model, loss_func, x, y_true, eps, e=0.00001):
    y_pred = model(x, use_sigmoid=False, use_tanh=True)
    grad_ = torch.autograd.grad(torch.sum(y_pred), x, retain_graph=True)[0]
    grad_norm =  torch.linalg.norm(grad_, dim=(1, 2)) ** 2
    coef_ = eps * y_pred / (grad_norm.reshape(-1, 1) + e)

    coef_ = coef_.unsqueeze(1).repeat(1, 50, 1)    
    perturb = - coef_ * grad_ 

    x_adv = x.data + perturb
    return x_adv


def fgsm_reg_attack(model, loss_func, x, y_true, eps, alpha):
    y_pred = model(x)
    loss_val = loss_func(y_pred, y_true)
    reg_value = reg_neigh(x, alpha)

    loss = loss_val - reg_value
    grad_ = torch.autograd.grad(loss, x, retain_graph=True)[0]
    x_adv = x.data + eps * (torch.sign(grad_))

    return x_adv

def only_disc_attack(
    model, 
    loss_func, 
    x, 
    y_true, 
    eps: float, 
    alpha: float, 
    disc_models: List, 
):
    reg_value = reg_disc(x, alpha, disc_models)
    grad_reg = torch.autograd.grad(reg_value, x, retain_graph=True)[0]

    grad_ = - grad_reg

    # print('grad loss', grad_loss[:3, :3].flatten())
    # print('grad reg', grad_reg[:3, :3].flatten())
    # print(torch.norm(grad_loss), torch.norm(grad_reg), torch.norm(grad_))
    x_adv = x.data + eps * torch.sign(grad_)

    return x_adv

def fgsm_disc_attack(
    model, 
    loss_func, 
    x, 
    y_true, 
    eps: float, 
    alpha: float, 
    disc_models: List, 
):
    y_pred = model(x)
    loss_val = loss_func(y_pred, y_true)
    grad_loss = torch.autograd.grad(loss_val, x, retain_graph=True)[0]
    reg_value = reg_disc(x, alpha, disc_models)
    grad_reg = torch.autograd.grad(reg_value, x, retain_graph=True)[0]

    grad_ = grad_loss - grad_reg

    # print('grad loss', grad_loss[:3, :3].flatten())
    # print('grad reg', grad_reg[:3, :3].flatten())
    # print(torch.norm(grad_loss), torch.norm(grad_reg), torch.norm(grad_))
    x_adv = x.data + eps * torch.sign(grad_)

    return x_adv


def reg_neigh(x, alpha):
    x_anchor = x[:, 1:-1]
    x_left = x[:, 2:]
    x_right = x[:, :-2]
    x_regular = (x_left + x_right) / 2
    reg_value = torch.sum((x_anchor - x_regular.detach()) ** 2, dim=list(range(1, len(x.shape))))
    reg_value = alpha * torch.mean(reg_value)
    return reg_value


def reg_disc(x, alpha: float, disc_models: List):
    n_models = len(disc_models)
    reg_value = 0
    for d_model in disc_models:
        req_grad(d_model, state=True)
        model_output = torch.mean(torch.log(F.sigmoid(d_model(x))))
        reg_value = reg_value + model_output

    reg_value = alpha* reg_value / n_models
    return reg_value