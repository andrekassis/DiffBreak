from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


def L2_norm(x, keepdim=False):
    z = (x**2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1] * (len(x.shape) - 1))
    return z


class APGDAttack(torch.nn.Module):
    """
    AutoPGD
    https://arxiv.org/abs/2003.01690

    :param predict:       forward pass function
    :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           bound on the norm of perturbations
    :param loss:          loss to optimize ('ce', 'dlr' supported)
    :param eot_iter:      iterations for Expectation over Trasformation
    :param rho:           parameter for decreasing the step size
    """

    def __init__(
        self,
        model,
        eot_iters=1,
        targeted=True,
        norm="Linf",
        eps=0.3,
        n_iter=100,
        n_restarts=1,
        rho=0.75,
        topk=None,
    ):
        """
        AutoPGD implementation in PyTorch
        """

        assert norm in ["Linf", "L2"]
        assert not eps is None

        super().__init__()
        self.model = model
        self.loss_fn = model.get_loss_fn()
        self.norm = norm
        self.eps = eps
        self.eot_iter = eot_iters
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.targeted = targeted

        self.thr_decr = rho
        self.topk = topk
        self.use_rs = True
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0.0
        self.criterion = self.loss_fn

        ### set parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    def eval(self):
        self.model = self.model.eval()
        return super().eval()

    def to(self, device):
        self.model = self.model.to(device)
        return super().to(device)

    def init_hyperparam(self, x):
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(x.device)
        for counter5 in range(k):
            t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == "Linf":
            t = x.abs().view(x.shape[0], -1).max(1)[0]
        else:
            t = (x**2).view(x.shape[0], -1).sum(-1).sqrt()

        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def init_perturbation(self, shape, device):
        fn = torch.rand if self.norm == "Linf" else torch.randn
        scale = lambda x: (2 * x - 1) if self.norm == "Linf" else x
        t = scale(fn(shape).to(device).detach())
        return self.eps * torch.ones(shape, device=device) * self.normalize(t)

    def step_linf(self, x_adv, x, grad, grad2, a, step_size):
        x_adv_1 = x_adv + step_size * torch.sign(grad)
        x_adv_1 = torch.clamp(
            torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps),
            0.0,
            1.0,
        )

        return torch.clamp(
            torch.min(
                torch.max(
                    x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                    x - self.eps,
                ),
                x + self.eps,
            ),
            0.0,
            1.0,
        )

    def step_l2(self, x_adv, x, grad, grad2, a, step_size):
        x_adv_1 = x_adv + step_size * self.normalize(grad)
        x_adv_1 = torch.clamp(
            x
            + self.normalize(x_adv_1 - x)
            * torch.min(
                self.eps * torch.ones_like(x).detach(),
                L2_norm(x_adv_1 - x, keepdim=True),
            ),
            0.0,
            1.0,
        )
        x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
        return torch.clamp(
            x
            + self.normalize(x_adv_1 - x)
            * torch.min(
                self.eps * torch.ones_like(x).detach(),
                L2_norm(x_adv_1 - x, keepdim=True),
            ),
            0.0,
            1.0,
        )

    def step(self, x_adv, x, grad, grad2, a, step_size):
        fn = self.step_linf if self.norm == "Linf" else self.step_l2
        return fn(x_adv, x, grad, grad2, a, step_size) + 0.0

    def prop(self, x_adv, y):
        x_adv.requires_grad_()
        grad = torch.zeros_like(x_adv)
        for ei in range(self.eot_iter):
            print(f"eot iter: {ei}")
            with torch.enable_grad():
                logits = self.model(x_adv)
                loss_indiv = self.criterion(logits, y)
                loss = loss_indiv.sum()
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            torch.cuda.empty_cache()
            gc.collect()
        grad /= float(self.eot_iter)
        return logits, loss_indiv, loss, grad

    def update_params(
        self,
        x_adv,
        grad,
        step_size,
        loss_steps,
        i,
        k,
        loss_best,
        reduced_last_check,
        loss_best_last_check,
        x_best,
        grad_best,
    ):
        fl_oscillation = self.check_oscillation(
            loss_steps, i, k, loss_best, k3=self.thr_decr
        )
        fl_reduce_no_impr = (1.0 - reduced_last_check) * (
            loss_best_last_check >= loss_best
        ).float()
        fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
        reduced_last_check = fl_oscillation.clone()
        loss_best_last_check = loss_best.clone()

        if fl_oscillation.sum() > 0:
            ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
            step_size[ind_fl_osc] /= 2.0

            x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
            grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

        return x_adv, grad, loss_best_last_check, reduced_last_check, step_size

    def attack_single_run(self, x, y):
        x_best_adv_single = None
        single_found = False
        k = self.n_iter_2
        n_fts = math.prod(self.orig_dim)
        counter, counter3 = 0, 0
        step_size = (
            2.0
            * self.eps
            * torch.ones([x.shape[0], *([1] * self.ndims)]).to(x.device).detach()
        )
        x_adv = (x + self.init_perturbation(x.shape, x.device)).clamp(0.0, 1.0)
        x_best, x_best_adv = x_adv.clone(), x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]).to(x.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]).to(x.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        print("******iteration 1*****")
        logits, loss_indiv, loss, grad = self.prop(x_adv, y)
        grad_best = grad.clone()

        acc = (
            logits.detach().max(1)[1] == y
            if not self.targeted
            else logits.detach().max(1)[1] != y
        ).float().mean(0, keepdims=True) > 0.5
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        x_adv_old = x_adv.clone()
        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

        for i in range(self.n_iter):
            print(f"******iteration {i+2}*****")
            with torch.no_grad():
                x_adv_old = x_adv.detach().clone()
                x_adv = self.step(
                    x_adv.detach(),
                    x,
                    grad,
                    x_adv.detach() - x_adv_old,
                    0.75 if i > 0 else 1.0,
                    step_size,
                )

            logits, loss_indiv, loss, grad = self.prop(x_adv, y)

            pred = (
                logits.detach().max(1)[1] == y
                if not self.targeted
                else logits.detach().max(1)[1] != y
            ).float().mean(0, keepdims=True) > 0.5

            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero().squeeze()
            x_best_adv = x_adv + 0.0

            ### check step size
            with torch.no_grad():
                loss_steps[i] = loss_indiv.detach().clone() + 0
                ind = (loss_indiv.detach().clone() > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = loss_indiv.detach().clone()[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0
                counter3 += 1

                if counter3 == k:
                    (
                        x_adv,
                        grad,
                        loss_best_last_check,
                        reduced_last_check,
                        step_size,
                    ) = self.update_params(
                        x_adv,
                        grad,
                        step_size,
                        loss_steps,
                        i,
                        k,
                        loss_best,
                        reduced_last_check,
                        loss_best_last_check,
                        x_best,
                        grad_best,
                    )
                    k = max(k - self.size_decr, self.n_iter_min)
                    counter3 = 0

            _, succ, single_succ = self.model.eval_attack(
                x_best_adv.detach(), y, targeted=self.targeted
            )
            if single_succ:
                single_found = True
                x_best_adv_single = x_best_adv.detach()
            if succ:
                break
        if single_found and not succ:
            x_best_adv = x_best_adv_single
        return x_best_adv, succ, float(single_found)

    def forward(self, x, y, y_init):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True the points attaining highest loss
                            are returned, otherwise adversarial examples
        """
        self.model.set_y_orig(y_init)
        self.init_hyperparam(x)
        y = y.detach().long()
        tot_single_succ = False
        for counter in range(self.n_restarts):
            adv, succ, single_succ = self.attack_single_run(x, y)
            if single_succ:
                single_adv = adv.detach()
                tot_single_succ = True
            if succ:
                break
        if tot_single_succ and not succ:
            adv = single_adv
        return adv, succ, single_succ
