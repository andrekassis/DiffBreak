import math
import logging
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from .wrapper import ClassifierWrapper
from ..diffusion import LossWrapper, RevDiffusion, DBPWrapper
from ..diffusion.diffusers import default_scaler, ModelWrapper
from .losses import *
from ..attacks import *
from .classifiers.torch import Classifier
from .logs import get_logger

logger = get_logger()


all_attacks = {
    "id": ID,
    "LF": LF,
    "apgd": APGDAttack,
    "diffattack_apgd": APGDAttack,
    "diffattack_LF": LF,
    "pgd": PGD,
    "stadv": StAdvAttack,
    "ppgd": PerceptualPGDAttack,
    "lagrange": LagrangePerceptualAttack,
}


def respaces_timesteps(steps, timestep_respacing, betas):
    if timestep_respacing != steps:
        assert timestep_respacing <= steps and steps % timestep_respacing == 0
        timestep_map = []
        frac_stride = (
            ((steps - 1) / (timestep_respacing - 1)) if timestep_respacing > 1 else 1
        )
        cur_idx, taken_steps = 0.0, []
        for _ in range(timestep_respacing):
            taken_steps.append(round(cur_idx))
            cur_idx += frac_stride
        orig_alphas_cumprod, last_alpha_cumprod, new_betas, use_timesteps = (
            np.cumprod(1 - betas, axis=0),
            1.0,
            [],
            set(taken_steps),
        )
        for i, alpha_cumprod in enumerate(orig_alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)
        betas = np.array(new_betas, dtype=betas.dtype)
    else:
        timestep_map = list(range(steps))
    return torch.tensor(betas), torch.tensor(timestep_map), len(betas)


def update_grad_mode(diffusion_args):
    grad_mode = diffusion_args.get("grad_mode", "full")
    assert grad_mode in [
        "full",
        "full_intermediate",
        "adjoint",
        "bpda",
        "blind",
        "forward_diff_only",
    ]
    grad_modes = {
        "with_intermediate": False,
        "basic_adjoint_method": False,
        "bpda": False,
        "blind": False,
        "forward_diff_only": False,
    }

    grad_mode_map = {
        "full_intermediate": "with_intermediate",
        "adjoint": "basic_adjoint_method",
        "bpda": "bpda",
        "blind": "blind",
        "forward_diff_only": "forward_diff_only",
    }
    if grad_mode != "full":
        grad_modes[grad_mode_map[grad_mode]] = True
    if "grad_mode" in list(diffusion_args.keys()):
        del diffusion_args["grad_mode"]

    diffusion_args.update(grad_modes)
    return diffusion_args


def update_diffusion_type(diffusion_args):
    diffusion_type = diffusion_args.get("diffusion_type", "vpsde")
    assert diffusion_type in ["vpsde", "vpode", "ddpm", "ddpm_ode"]

    d_type = (
        "vpsde"
        if diffusion_type == "vpode"
        else ("ddpm" if diffusion_type == "ddpm_ode" else diffusion_type)
    )

    d_type = {"diffusion_type": d_type, "ode": diffusion_type in ["vpode", "ddpm_ode"]}

    diffusion_args.update(d_type)
    return diffusion_args


def construct_diffusion_args(
    diffusion_type="vpsde",
    deterministic=False,
    grad_mode="full",
    grad_powers=None,
):
    if grad_powers is None:
        grad_powers = (0.0, 0.0)
    assert isinstance(grad_powers, tuple) and len(grad_powers) == 2
    assert isinstance(grad_powers[0], float) or isinstance(grad_powers[0], int)
    assert isinstance(grad_powers[1], float) or isinstance(grad_powers[1], int)
    assert isinstance(deterministic, bool)
    diffusion_args = {
        "deterministic": deterministic,
        "diffusion_type": diffusion_type,
        "grad_mode": grad_mode,
        "power": grad_powers[0],
        "power_2": grad_powers[1],
    }
    diffusion_args = update_grad_mode(diffusion_args)
    diffusion_args = update_diffusion_type(diffusion_args)
    return diffusion_args


def construct_diffusion_config(
    diffusion_steps=100,
    diffusion_repeats=1,
    beta_min=0.1,
    beta_max=20.0,
    full_horizon_length=1000,
    diffusion_t_mode="same",
    timestep_respacing=None,
    deterministic=False,
    batch_size=None,
):
    if deterministic:
        if batch_size is not None and batch_size != 1:
            logger.warning(
                "For deterministic dbp, you can only use batch_size=1. This setting will "
                "be changed accordingly"
            )
        batch_size = 1
    assert isinstance(batch_size, int) and batch_size > 0

    assert isinstance(beta_min, float) and beta_min >= 0
    assert isinstance(beta_max, float) and beta_max >= beta_min
    assert int(10 ** math.ceil(math.log10(full_horizon_length))) == full_horizon_length
    assert diffusion_t_mode in ["same", "plus1"]
    assert timestep_respacing is None or (
        isinstance(timestep_respacing, int)
        and timestep_respacing > 0
        and timestep_respacing <= full_horizon_length
    )

    timestep_respacing = (
        full_horizon_length if timestep_respacing is None else timestep_respacing
    )

    precision = len(str(full_horizon_length)) - 1
    betas = np.linspace(
        beta_min / full_horizon_length,
        beta_max / full_horizon_length,
        full_horizon_length,
        dtype=np.float64,
    )
    betas, timestep_map, num_diffusion_timesteps = respaces_timesteps(
        full_horizon_length, timestep_respacing, betas
    )
    preprocess_diffusion_t = lambda t: ((t + 1) if diffusion_t_mode == "plus1" else t)

    return {
        "steps": diffusion_steps,
        "num_diffusion_timesteps": num_diffusion_timesteps,
        "iterations": diffusion_repeats,
        "betas": betas,
        "timestep_map": timestep_map,
        "precision": precision,
        "preprocess_diffusion_t": preprocess_diffusion_t,
        "batch_size": batch_size,
    }


def construct_conditioning(
    guidance_mode=None,
    guidance_scale=60000.0,
    guidance_ptb=8.0,
):
    if guidance_mode is not None:
        assert guidance_mode in ["MSE", "SSIM"]
        assert (
            isinstance(guidance_scale, float) or isinstance(guidance_scale, int)
        ) and guidance_scale >= 0
        assert (
            isinstance(guidance_ptb, float) or isinstance(guidance_ptb, int)
        ) and guidance_ptb >= 0

    return {
        "guide_mode": guidance_mode,
        "guide_scale": guidance_scale,
        "ptb": guidance_ptb,
    }


def construct_dm(
    dm_class,
    image_size,
    effective_size,
    diffusion_config,
    conditioning,
):
    dm = dm_class(effective_size)
    dm, scaler = dm if isinstance(dm, tuple) else (dm, default_scaler)
    if not isinstance(dm, ModelWrapper):
        logger.error(f"arg0 must be a subclass of DMWrapper")
        exit(1)

    return RevDiffusion(
        dm,
        scaler,
        diffusion_config["steps"],
        diffusion_config["betas"],
        diffusion_config["timestep_map"],
        image_size,
        effective_size,
        num_diffusion_timesteps=diffusion_config["num_diffusion_timesteps"],
        precision=diffusion_config["precision"],
        preprocess_diffusion_t=diffusion_config["preprocess_diffusion_t"],
        batch_size=diffusion_config["batch_size"],
        cond_args=conditioning,
    )


def handle_diffattack(system):
    if not isinstance(system, DBPWrapper):
        logger.error("cannot use diffattack on non-DBP systems")
        exit(1)
    if (
        (not system.dm.with_intermediate)
        or system.dm.basic_adjoint_method
        or system.dm.bpda
        or system.dm.blind
        or system.dm.forward_diff_only
    ):
        logger.warning(
            "You are attempting to use diffattack with a gradient method that is not supported "
            "(only full_intermediate is allowed). I'm assuming you intend to use the correct method "
            "and changing the gradient method accordingly"
        )
        system.dm.basic_adjoint_method = False
        system.dm.bpda = False
        system.dm.blind = False
        system.dm.forward_diff_only = False
        system.dm.with_intermediate = True

        logger.warning(
            "Using an equal loss factor for all intermediate steps (default for the diffattack method). "
            "If you wish to change this behavior, provide 'power' keyword argument to Initializer.dbp()"
        )

    system.orig_loss = DiffAttack(
        system.orig_loss,
        t_interval=1,
        negate=system.orig_loss.negate,
        targeted=system.orig_loss.targeted,
    )
    system.dm.model_loss = LossWrapper(system.orig_loss, system.dm.dm)
    return system


def is_negated_loss(attack_name, loss_fn):
    if "pgd" in attack_name or attack_name == "lagrange":
        return isinstance(loss_fn, CE)
    return not isinstance(loss_fn, CE)


class Loader(DataLoader):
    def __init__(self, dataset, image_size, total_samples, **kwargs):
        super().__init__(dataset, **kwargs)
        self.image_size = image_size
        self.num_total_samples = total_samples


class Initializer:
    @staticmethod
    def data(data, total_samples=256, balanced_splits=False):
        data.setup(total_samples=total_samples, balanced_splits=balanced_splits)
        image_size = data.crop_sz if data.crop_sz is not None else data.target_size
        ##make sure to take iter
        data = Loader(data, image_size, total_samples, batch_size=1, shuffle=True)
        return data

    @staticmethod
    def dbp(
        dm_class,
        image_size,
        batch_size=None,
        diffusion_type="vpsde",
        diffusion_steps=None,
        diffusion_repeats=1,
        deterministic=False,
        grad_mode="full",
        grad_powers=None,
        beta_min=0.1,
        beta_max=20.0,
        full_horizon_length=1000,
        diffusion_t_mode="same",
        timestep_respacing=None,
        guidance_mode=None,
        guidance_scale=60000.0,
        guidance_ptb=8.0,
    ):
        next_power_of_2 = lambda x: 1 if x == 0 else 2 ** (x - 1).bit_length()
        effective_size = next_power_of_2(image_size)

        if grad_mode == "adjoint":
            if diffusion_type not in ["vpsde", "vpode"]:
                logger.error("Can't use adjoint with non-vp-based dbp.")
                exit(1)

        if batch_size is None:
            logger.error("parameter batch_size must be provided for DBP")
            exit(1)

        if diffusion_steps is None:
            logger.error("parameter diffusion_steps must be provided for DBP")
            exit(1)
        assert isinstance(batch_size, int) and batch_size > 0
        assert isinstance(diffusion_steps, int) and diffusion_steps > 0
        if "ddpm" in diffusion_type:
            if guidance_mode is None:
                logger.info(
                    "You are using vanilla DDPM-based DBP without guidance. To use guidance, "
                    "provide guidance_mode, guidance_scale and guidance_ptb arguments"
                )
        if "ddpm" not in diffusion_type and guidance_mode is not None:
            logger.warning(
                "You provided guidance_mode argument for a vp-based DBP scheme that currently "
                "doesn't support guidance. This argument will be ignored"
            )

        diffusion_config = construct_diffusion_config(
            diffusion_steps=diffusion_steps,
            diffusion_repeats=diffusion_repeats,
            beta_min=beta_min,
            beta_max=beta_max,
            full_horizon_length=full_horizon_length,
            diffusion_t_mode=diffusion_t_mode,
            timestep_respacing=timestep_respacing,
            deterministic=deterministic,
            batch_size=batch_size,
        )

        diffusion_args = construct_diffusion_args(
            diffusion_type=diffusion_type,
            deterministic=deterministic,
            grad_mode=grad_mode,
            grad_powers=grad_powers,
        )

        conditioning = construct_conditioning(
            guidance_mode=guidance_mode,
            guidance_scale=guidance_scale,
            guidance_ptb=guidance_ptb,
        )

        dm = construct_dm(
            dm_class,
            image_size,
            effective_size,
            diffusion_config,
            conditioning,
        )

        diffusion_args.update(
            {
                "dm": dm,
                "diffusion_steps": diffusion_config["steps"],
                "diffusion_iterations": diffusion_config["iterations"],
                "eval_batch_sz": diffusion_config["batch_size"],
            }
        )
        return diffusion_args

    @staticmethod
    def defended_classifier(
        classifier, loss_fn=None, eval_mode="batch", verbose=0, dbp=None
    ):
        assert isinstance(eval_mode, str) and eval_mode in ["single", "batch"]
        assert isinstance(verbose, int) and verbose >= 0 and verbose <= 2

        if not isinstance(classifier, Classifier):
            from .classifiers.tf import Classifier as TFClassifier

            if not isinstance(classifier, TFClassifier):
                logger.error(
                    "arg0 must be either an instance of PyTorchClassifier "
                    "or PyTorchClassifier"
                )

        eval_mode = "single" if dbp is None else eval_mode
        constructor = ClassifierWrapper if dbp is None else DBPWrapper
        dbp = dbp if dbp is not None else {}
        assert isinstance(dbp, dict)

        if loss_fn is not None:
            if not (
                isinstance(loss_fn, CE)
                or isinstance(loss_fn, MarginLoss)
                or isinstance(loss_fn, DLR)
            ):
                logger.error(
                    "Supported loss functions are currently only: CE, MarginLoss, DLR. "
                    "Please refer to docs."
                )
                exit(1)
        return constructor(
            classifier, loss_fn, eval_mode=eval_mode, verbose=verbose, **dbp
        )

    @staticmethod
    def attack(system, targeted=False, **attack_params):
        attack_name = attack_params["attack_name"]
        del attack_params["attack_name"]
        assert isinstance(system, ClassifierWrapper)
        if attack_name not in list(all_attacks.keys()):
            logger.error(
                f"Supported attacks are currently only {list(all_attacks.keys())}"
            )
            exit(1)

        if attack_name != "id":
            assert system.orig_loss is not None

        loss_fn = system.orig_loss

        if "diffattack" in attack_name:
            system = handle_diffattack(system)

        if system.orig_loss is not None:
            system.orig_loss.is_targeted(targeted)
            system.orig_loss.is_negated(is_negated_loss(attack_name, loss_fn))

        eot_iters = attack_params.get("eot_iters", 1)
        assert isinstance(eot_iters, int) and eot_iters >= 1
        if not isinstance(system, DBPWrapper) or system.dm.deterministic:
            if eot_iters != 1:
                logger.warning(
                    f"You've provided eot_iters={eot_iters} for a deterministic defense. "
                    f"This will be overwrittin with eot_iters=1"
                )
            eot_iters = 1
        attack_params["eot_iters"] = eot_iters
        attack_params["targeted"] = targeted

        return all_attacks[attack_name](system, **attack_params)
