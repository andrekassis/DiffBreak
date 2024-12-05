import torch
import numpy as np
import gc
import logging

from .losses import MarginLoss
from .classifiers.torch.classifier import Classifier
from .logs import get_logger

logger = get_logger()


class ClassifierWrapper(torch.nn.Module):
    def __init__(
        self,
        model_fn,
        model_loss,
        eval_mode="batch",
        verbose=0,
    ):
        super().__init__()

        if not isinstance(model_fn, Classifier):
            from .classifiers.tf.classifier import (
                Classifier as TFClassifier,
            )

            if not isinstance(model_fn, TFClassifier):
                logger.error(
                    "Provided classifier should be either a subclass of utils.classifiers.torch.classifier "
                    "or utils.classifiers.tf.classifier"
                )
                exit(1)

        model_fn.requires_grad_(False)
        self.clip_min = 0.0
        self.clip_max = 1.0
        self.model_fn = model_fn
        self.orig_loss = model_loss
        self.eval_mode = eval_mode
        self.verbose = verbose

    def to(self, device):
        self.model_fn = self.model_fn.to(device)
        return super().to(device)

    def eval(self):
        self.model_fn.eval()
        return super().eval()

    def forward(self, x, steps=None):
        return self.model_fn(self.preprocess_forward(x, steps=steps))

    def get_loss_fn(self):
        return self.orig_loss

    def set_y_orig(self, y):
        if self.orig_loss is not None:
            self.orig_loss.y_orig = torch.argmax(y, dim=-1)

    def reset_power(self):
        return

    def preprocess_eval(self, x):
        return x.clone()

    def preprocess_forward(self, x, steps=None):
        return x

    def propagate(self, x):
        with torch.no_grad():
            x_ev = self.preprocess_eval(x)
            return self.model_fn(x_ev.clamp(self.clip_min, self.clip_max))

    def eval_attack(self, x, label, targeted=True):
        # assert self.orig_loss is not None
        with torch.no_grad():
            x_ev = self.preprocess_eval(x)
            # logits_ev = self.propagate(x)
            logits_ev = self.model_fn(x_ev.clamp(self.clip_min, self.clip_max))
            if self.orig_loss is not None and self.verbose == 2:
                label_loss_ev = self.orig_loss(logits_ev, label)
                logger.info(f"loss: {label_loss_ev.mean().detach().item()}")

            logit_score_ev = MarginLoss(kappa="inf", targeted=True, negate=True)(
                labels=label, logits=logits_ev.detach(), do_mean=False
            ).squeeze()
            success_rate = (
                (logit_score_ev < 0).float().mean()
                if targeted
                else (logit_score_ev > 0).float().mean()
            )
            successful_attack_single = success_rate > 0
            if self.eval_mode == "single":
                successful_attack = success_rate > 0
            else:
                uniq, counts = torch.unique(
                    torch.argmax(logits_ev, dim=-1), return_counts=True
                )
                label_idx = (
                    torch.where(uniq == label.detach().item())[0].detach().squeeze()
                )
                if len(label_idx.shape) == 0:
                    label_idx = label_idx.item()
                    successful_attack = (
                        (torch.argmax(counts).detach().item() == label_idx)
                        if targeted
                        else (torch.argmax(counts).detach().item() != label_idx)
                    )
                else:
                    successful_attack = not targeted
            if self.verbose == 2:
                logger.info(f"scores: {logit_score_ev.detach().cpu().numpy()}")

        # del x_ev
        torch.cuda.empty_cache()
        gc.collect()
        return success_rate, float(successful_attack), float(successful_attack_single)

    def eval_sample(self, x, y_init):
        if self.orig_loss is not None:
            y_o = self.orig_loss.y_orig if hasattr(self.orig_loss, "y_orig") else None
            self.orig_loss.y_orig = torch.argmax(y_init, dim=-1)
        init_success_rate, init_score, _ = self.eval_attack(
            x,
            torch.argmax(y_init, dim=-1),
            targeted=True,
        )
        if self.orig_loss is not None:
            self.orig_loss.y_orig = y_o
        return init_success_rate, init_score
