import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .utils import *


def normalize_flatten_features(
    features,
    eps=1e-10,
):
    """
    Given a tuple of features (layer1, layer2, layer3, ...) from a network,
    flattens those features into a single vector per batch input. The
    features are also scaled such that the L2 distance between features
    for two different inputs is the LPIPS distance between those inputs.
    """

    normalized_features = []
    for feature_layer in features:
        norm_factor = torch.sqrt(torch.sum(feature_layer**2, dim=1, keepdim=True)) + eps
        normalized_features.append(
            (
                feature_layer
                / (
                    norm_factor
                    * np.sqrt(feature_layer.size()[2] * feature_layer.size()[3])
                )
            ).view(feature_layer.size()[0], -1)
        )
    return torch.cat(normalized_features, dim=1)


class BisectionPerceptualProjection(nn.Module):
    def __init__(self, bound, lpips_model, num_steps=10):
        super().__init__()

        self.bound = bound
        self.lpips_model = lpips_model
        self.num_steps = num_steps

    def forward(self, inputs, adv_inputs, input_features=None):
        batch_size = inputs.shape[0]
        if input_features is None:
            input_features = normalize_flatten_features(
                self.lpips_model.features(inputs)
            )

        lam_min = torch.zeros(batch_size, device=inputs.device)
        lam_max = torch.ones(batch_size, device=inputs.device)
        lam = 0.5 * torch.ones(batch_size, device=inputs.device)

        for _ in range(self.num_steps):
            projected_adv_inputs = (
                inputs * (1 - lam[:, None, None, None])
                + adv_inputs * lam[:, None, None, None]
            )
            adv_features = self.lpips_model.features(projected_adv_inputs)
            adv_features = normalize_flatten_features(adv_features).detach()
            diff_features = adv_features - input_features
            norm_diff_features = torch.norm(diff_features, dim=1)

            lam_max[norm_diff_features > self.bound] = lam[
                norm_diff_features > self.bound
            ]
            lam_min[norm_diff_features <= self.bound] = lam[
                norm_diff_features <= self.bound
            ]
            lam = 0.5 * (lam_min + lam_max)
        return projected_adv_inputs.detach()


class NewtonsPerceptualProjection(nn.Module):
    def __init__(
        self, bound, lpips_model, projection_overshoot=1e-1, max_iterations=10
    ):
        super().__init__()

        self.bound = bound
        self.lpips_model = lpips_model
        self.projection_overshoot = projection_overshoot
        self.max_iterations = max_iterations
        self.bisection_projection = BisectionPerceptualProjection(bound, lpips_model)

    def forward(self, inputs, adv_inputs, input_features=None):
        original_adv_inputs = adv_inputs
        if input_features is None:
            input_features = normalize_flatten_features(
                self.lpips_model.features(inputs)
            )

        needs_projection = torch.ones_like(adv_inputs[:, 0, 0, 0]).bool()

        needs_projection.requires_grad = False
        iteration = 0
        while needs_projection.sum() > 0 and iteration < self.max_iterations:
            adv_inputs.requires_grad = True
            adv_features = normalize_flatten_features(
                self.lpips_model.features(adv_inputs[needs_projection])
            )
            adv_lpips = (input_features[needs_projection] - adv_features).norm(dim=1)
            adv_lpips.sum().backward()

            projection_step_size = (adv_lpips - self.bound).clamp(min=0)
            projection_step_size[projection_step_size > 0] += self.projection_overshoot

            grad_norm = (
                adv_inputs.grad.data[needs_projection]
                .view(needs_projection.sum(), -1)
                .norm(dim=1)
            )
            inverse_grad = (
                adv_inputs.grad.data[needs_projection]
                / grad_norm[:, None, None, None] ** 2
            )

            adv_inputs.data[needs_projection] = (
                (
                    adv_inputs.data[needs_projection]
                    - projection_step_size[:, None, None, None]
                    * (1 + self.projection_overshoot)
                    * inverse_grad
                )
                .clamp(0, 1)
                .detach()
            )

            needs_projection[needs_projection.clone()] = projection_step_size > 0
            iteration += 1

        if needs_projection.sum() > 0:
            # If we still haven't projected all inputs after max_iterations,
            # just use the bisection method.
            adv_inputs = self.bisection_projection(
                inputs, original_adv_inputs, input_features
            )

        return adv_inputs.detach()


class FirstOrderStepPerceptualAttack(nn.Module):
    def __init__(
        self,
        model,
        loss_fn,
        bound=0.5,
        num_iterations=5,
        h=1e-3,
        lpips_model="self",
        targeted=False,
        include_image_as_activation=False,
    ):
        """
        Perceptual attack using conjugate gradient to solve the constrained
        optimization problem.

        bound is the (approximate) bound on the LPIPS distance.
        num_iterations is the number of CG iterations to take.
        h is the step size to use for finite-difference calculation.
        """

        super().__init__()

        self.model = model
        self.bound = bound
        self.num_iterations = num_iterations
        self.h = h

        self.lpips_model = get_lpips_model(lpips_model, model)
        self.lpips_distance = LPIPSDistance(
            self.lpips_model,
            include_image_as_activation=include_image_as_activation,
        )
        self.loss = loss_fn

    def _multiply_matrix(self, v):
        """
        If (D phi) is the Jacobian of the features function for the model
        at inputs, then approximately calculates
            (D phi)T (D phi) v
        """

        self.inputs.grad.data.zero_()

        with torch.no_grad():
            v_features = self.lpips_model.features(self.inputs.detach() + self.h * v)
            D_phi_v = (
                normalize_flatten_features(v_features) - self.input_features
            ) / self.h

        torch.sum(self.input_features * D_phi_v).backward(retain_graph=True)
        return self.inputs.grad.data.clone()

    def forward(self, inputs, labels, eot_iters=1):
        self.inputs = inputs
        inputs.requires_grad = True
        if self.model == self.lpips_model:
            input_features, orig_logits = self.model.features_logits(inputs)
        else:
            input_features = self.lpips_model.features(inputs)
            orig_logits = self.model(inputs)
        self.input_features = normalize_flatten_features(input_features)

        loss = self.loss(orig_logits, labels)
        loss.sum().backward(retain_graph=True)
        inputs_grad = inputs.grad.data.clone()

        if inputs_grad.abs().max() < 1e-4:
            return inputs

        x = torch.zeros_like(inputs)
        r = inputs_grad - self._multiply_matrix(x)
        p = r

        for cg_iter in range(self.num_iterations):
            r_last = r
            p_last = p
            x_last = x
            del r, p, x

            r_T_r = (r_last**2).sum(dim=[1, 2, 3])
            if r_T_r.max() < 1e-1 and cg_iter > 0:
                # If the residual is small enough, just stop the algorithm.
                x = x_last
                break

            A_p_last = self._multiply_matrix(p_last)
            # print('|r|^2 =', ' '.join(f'{z:.2f}' for z in r_T_r))
            alpha = (r_T_r / (p_last * A_p_last).sum(dim=[1, 2, 3]))[
                :, None, None, None
            ]
            x = x_last + alpha * p_last

            # These calculations aren't necessary on the last iteration.
            if cg_iter < self.num_iterations - 1:
                r = r_last - alpha * A_p_last

                beta = ((r**2).sum(dim=[1, 2, 3]) / r_T_r)[:, None, None, None]
                p = r + beta * p_last

        x_features = self.lpips_model.features(self.inputs.detach() + self.h * x)
        D_phi_x = (
            normalize_flatten_features(x_features) - self.input_features
        ) / self.h

        lam = (self.bound / D_phi_x.norm(dim=1))[:, None, None, None]

        inputs_grad_norm = inputs_grad.reshape(inputs_grad.size()[0], -1).norm(dim=1)
        # If the grad is basically 0, don't perturb that input. It's likely
        # already misclassified, and trying to perturb it further leads to
        # numerical instability.
        lam[inputs_grad_norm < 1e-4] = 0
        x[inputs_grad_norm < 1e-4] = 0

        return lam * x  # (inputs + lam * x).clamp(0, 1).detach()


class PerceptualPGDAttack(nn.Module):
    def __init__(
        self,
        model,
        bound=0.5,
        step=None,
        num_iterations=5,
        eot_iters=1,
        cg_iterations=5,
        h=1e-3,
        lpips_model="self",
        decay_step_size=False,
        targeted=False,
        num_classes=None,
        include_image_as_activation=False,
    ):
        """
        Iterated version of the conjugate gradient attack.

        step_size is the step size in LPIPS distance.
        num_iterations is the number of steps to take.
        cg_iterations is the conjugate gradient iterations per step.
        h is the step size to use for finite-difference calculation.
        project is whether or not to project the perturbation into the LPIPS
            ball after each step.
        """

        super().__init__()
        loss_fn = model.get_loss_fn()
        self.model = model
        self.bound = bound
        self.num_iterations = num_iterations
        self.decay_step_size = decay_step_size
        self.step = step
        self.targeted = targeted
        self.num_classes = num_classes
        self.eot_iters = eot_iters

        if self.step is None:
            if self.decay_step_size:
                self.step = self.bound
            else:
                self.step = 2 * self.bound / self.num_iterations

        self.lpips_model = get_lpips_model(lpips_model, model)
        self.first_order_step = FirstOrderStepPerceptualAttack(
            model,
            loss_fn,
            bound=self.step,
            num_iterations=cg_iterations,
            h=h,
            lpips_model=self.lpips_model,
            include_image_as_activation=include_image_as_activation,
            targeted=self.targeted,
        )
        self.projection = NewtonsPerceptualProjection(self.bound, self.lpips_model)

    def _attack(self, inputs, labels):
        best_adv_single = None
        single_found = False

        with torch.no_grad():
            input_features = normalize_flatten_features(
                self.lpips_model.features(inputs)
            )

        start_perturbations = torch.zeros_like(inputs)
        start_perturbations.normal_(0, 0.01)
        adv_inputs = inputs + start_perturbations
        for attack_iter in range(self.num_iterations):
            if self.decay_step_size:
                step_size = self.step * 0.1 ** (attack_iter / self.num_iterations)
                self.first_order_step.bound = step_size
            g = 0
            for eot_iter in range(self.eot_iters):
                g = g + self.first_order_step(adv_inputs.detach(), labels)
                torch.cuda.empty_cache()
                gc.collect()
            g = g / self.eot_iters
            adv_inputs = (adv_inputs + g).clamp(0, 1).detach()

            adv_inputs = self.projection(inputs, adv_inputs, input_features)
            _, succ, single_succ = self.model.eval_attack(
                adv_inputs.detach(), labels, targeted=self.targeted
            )
            if single_succ:
                single_found = True
                best_adv_single = adv_inputs.detach()
            if succ:
                break
        if single_found and not succ:
            adv_inputs = best_adv_single
        return adv_inputs, succ, float(single_found)

    def eval(self):
        self.lpips_model = self.lpips_model.eval()
        self.model = self.model.eval()
        return super().eval()

    def to(self, device):
        self.lpips_model = self.lpips_model.to(device)
        self.model = self.model.to(device)
        return super().to(device)

    def forward(self, inputs, labels, y_init):
        self.model.set_y_orig(y_init)
        return self._attack(inputs, labels)


class LagrangePerceptualAttack(nn.Module):
    def __init__(
        self,
        model,
        bound=0.5,
        step=None,
        num_iterations=20,
        eot_iters=1,
        binary_steps=5,
        h=0.1,
        lpips_model="self",
        decay_step_size=True,
        num_classes=None,
        include_image_as_activation=False,
        targeted=False,
    ):
        """
        Perceptual attack using a Lagrangian relaxation of the
        LPIPS-constrainted optimization problem.
        bound is the (soft) bound on the LPIPS distance.
        step is the LPIPS step size.
        num_iterations is the number of steps to take.
        lam is the lambda value multiplied by the regularization term.
        h is the step size to use for finite-difference calculation.
        lpips_model is the model to use to calculate LPIPS or 'self' or
            'alexnet'
        """

        super().__init__()
        loss_fn = model.get_loss_fn()
        self.model = model
        self.bound = bound
        self.decay_step_size = decay_step_size
        self.num_iterations = num_iterations
        self.eot_iters = eot_iters

        best_adv_single = None
        single_found = False

        if step is None:
            if self.decay_step_size:
                self.step = self.bound
            else:
                self.step = self.bound * 2 / self.num_iterations
        else:
            self.step = step
        self.binary_steps = binary_steps
        self.h = h
        self.targeted = targeted
        self.num_classes = num_classes

        self.lpips_model = get_lpips_model(lpips_model, model)
        self.lpips_distance = LPIPSDistance(
            self.lpips_model,
            include_image_as_activation=include_image_as_activation,
        )
        self.loss = loss_fn
        self.projection = NewtonsPerceptualProjection(self.bound, self.lpips_model)

    def threat_model_contains(self, inputs, adv_inputs):
        """
        Returns a boolean tensor which indicates if each of the given
        adversarial examples given is within this attack's threat model for
        the given natural input.
        """

        return self.lpips_distance(inputs, adv_inputs) <= self.bound

    def _attack(self, inputs, labels):
        perturbations = torch.zeros_like(inputs)
        perturbations.normal_(0, 0.01)
        perturbations.requires_grad = True

        batch_size = inputs.shape[0]
        step_size = self.step

        lam = 0.01 * torch.ones(batch_size, device=inputs.device)

        input_features = normalize_flatten_features(
            self.lpips_model.features(inputs)
        ).detach()

        live = torch.ones(batch_size, device=inputs.device, dtype=torch.bool)

        for binary_iter in range(self.binary_steps):
            for attack_iter in range(self.num_iterations):
                if self.decay_step_size:
                    step_size = self.step * (0.1 ** (attack_iter / self.num_iterations))
                else:
                    step_size = self.step

                tg = 0
                for eot_iter in range(self.eot_iters):
                    if perturbations.grad is not None:
                        perturbations.grad.data.zero_()

                    adv_inputs = (inputs + perturbations)[live]

                    if self.model == self.lpips_model:
                        adv_features, adv_logits = self.model.features_logits(
                            adv_inputs
                        )
                    else:
                        adv_features = self.lpips_model.features(adv_inputs)
                        adv_logits = self.model(adv_inputs)

                    adv_labels = adv_logits.argmax(1)
                    adv_loss = self.loss(adv_logits, labels[live])
                    adv_features = normalize_flatten_features(adv_features)
                    lpips_dists = (adv_features - input_features[live]).norm(dim=1)
                    all_lpips_dists = torch.zeros(batch_size, device=inputs.device)
                    all_lpips_dists[live] = lpips_dists

                    loss = -adv_loss + lam[live] * F.relu(lpips_dists - self.bound)
                    loss.sum().backward()

                    grad = perturbations.grad.data[live]
                    grad_normed = grad / (
                        grad.reshape(grad.size()[0], -1).norm(dim=1)[
                            :, None, None, None
                        ]
                        + 1e-8
                    )

                    dist_grads = (
                        adv_features
                        - normalize_flatten_features(
                            self.lpips_model.features(adv_inputs - grad_normed * self.h)
                        )
                    ).norm(dim=1) / self.h
                    tg = (
                        tg
                        + grad_normed
                        * (step_size / (dist_grads + 1e-8))[:, None, None, None]
                    )
                    torch.cuda.empty_cache()
                    gc.collect()

                updates = -tg / self.eot_iters

                perturbations.data[live] = (
                    (inputs[live] + perturbations[live] + updates).clamp(0, 1)
                    - inputs[live]
                ).detach()

                adv_inps = self.projection(
                    inputs, (inputs + perturbations).detach(), input_features
                )
                _, succ, single_succ = self.model.eval_attack(
                    adv_inps.detach(),
                    labels,
                    targeted=self.targeted,
                )

                if single_succ:
                    single_found = True
                    best_adv_single = adv_inps.detach()

            lam[all_lpips_dists >= self.bound] *= 10

        adv_inputs = (inputs + perturbations).detach()
        adv_inputs = self.projection(inputs, adv_inputs, input_features)

        if single_found and not succ:
            adv_inputs = best_adv_single

        return adv_inputs, succ, float(single_found)

    def eval(self):
        self.lpips_model = self.lpips_model.eval()
        self.model = self.model.eval()
        return super().eval()

    def to(self, device):
        self.lpips_model = self.lpips_model.to(device)
        self.model = self.model.to(device)
        return super().to(device)

    def forward(self, inputs, labels, y_init):
        self.model.set_y_orig(y_init)
        return self._attack(inputs, labels)
