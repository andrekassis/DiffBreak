import logging

from .data import *
from ..diffusion.diffusers import *
from .classifiers.torch import *
from .losses import *
from .logs import get_logger

logger = get_logger()

id_params = {}

stadv_params = {
    "bound": 0.05,
    "num_iterations": 100,
    "lr": 0.01,
}

LF_params = {
    "dist_fn": {"class": "LpipsVGG"},
    "eps": 5.0e-2,
    "max_iterations": 100,
    "filter_args": {
        "loss_factor": 0.0,
        "sigma_color": 0.05,
    },
    "optimizer_args": {
        "grad_sgn": False,
        "regularization": {"type": "l2", "factor": 0.0},
        "learning_rate": {"values": [0.05, 0.008]},
        "abort_early": True,
    },
}

apgd_params = {
    "norm": "Linf",
    "n_iter": 100,
    "n_restarts": 1,
}

lagrange_params = {
    "bound": 0.5,
    "num_iterations": 20,
    "binary_steps": 5,
}

ppgd_params = {
    "bound": 0.5,
    "num_iterations": 100,
    "cg_iterations": 5,
}

pgd_params = {"eps_iter": 0.008, "nb_iter": 100}


class Registry:
    all_datasets = ["cifar10", "celeba-hq", "imagenet", "youtube"]
    dbp_schemes = ["vpsde", "vpode", "ddpm", "ddpm_ode"]

    datamap = {
        "imagenet": ImageNetData,
        "cifar10": Cifar10Data,
        "celeba-hq": CelebAHQData,
        "youtube": YouTubeData,
    }

    dm_map = {
        "imagenet": GuidedModel,
        "celeba-hq": DDPMModel,
        "cifar10_vp": ScoreSDEModel,
        "cifar10_ddpm": HaoDDPM,
        "youtube": DDPMModel,
    }

    cifar10_classifiers = [
        "WIDERESNET_28_10",
        "WRN_28_10_AT0",
        "WRN_28_10_AT1",
        "WRN_70_16_AT0",
        "WRN_70_16_AT1",
        "WRN_70_16_L2_AT1",
        "WIDERESNET_70_16",
        "RESNET_50",
        "WRN_70_16_DROPOUT",
        "VGG16",
    ]

    imagenet_classifiers = [
        "RESNET18",
        "RESNET50",
        "RESNET101",
        "WIDERESNET_50_2",
        "DEIT_S",
    ]

    celeba_classifiers = ["NET_BEST"]
    youtube_classifiers = ["RESNET50NP"]

    all_classifiers = {
        "cifar10": cifar10_classifiers,
        "imagenet": imagenet_classifiers,
        "celeba-hq": celeba_classifiers,
        "youtube": youtube_classifiers,
    }

    default_classifiers = {
        "cifar10": "WIDERESNET_70_16",
        "imagenet": "DEIT_S",
        "celeba-hq": "NET_BEST",
        "youtube": "RESNET50NP",
    }

    dm_names = {
        "imagenet": "GuidedModel",
        "celeba-hq": "DDPMModel",
        "cifar10_vp": "ScoreSDEModel",
        "cifar10_ddpm": "HaoDDPM",
        "youtube": "DDPMModel",
    }

    grad_modes = [
        "full",
        "full_intermediate",
        "adjoint",
        "bpda",
        "blind",
        "forward_diff_only",
    ]

    all_attacks = [
        "pgd",
        "ppgd",
        "apgd",
        "diffattack_apgd",
        "LF",
        "diffattack_LF",
        "stadv",
        "lagrange",
        "id",
    ]

    @staticmethod
    def available_datasets():
        return Registry.all_datasets

    def __check_dataset(dataset_name):
        assert isinstance(dataset_name, str)
        if not dataset_name in Registry.available_datasets():
            logger.error(
                f"The registry does not contain default datasets, classifiers, dms or attacks"
                f" for dataset {dataset_name}. However, you can still implement your own. Please"
                f" refer to the docs."
            )
            exit(1)

    @staticmethod
    def available_classifiers(dataset_name):
        Registry.__check_dataset(dataset_name)
        return Registry.all_classifiers[dataset_name]

    @staticmethod
    def available_dms(dataset_name, diffusion_type="vpsde"):
        Registry.__check_dataset(dataset_name)
        if "vp" in diffusion_type and dataset_name == "cifar10":
            dataset_name = "cifar10_vp"
        elif dataset_name == "cifar10":
            dataset_name = "cifar10_ddpm"
        return Registry.dm_names[dataset_name]

    @staticmethod
    def available_dbp_schemes():
        return Registry.dbp_schemes

    @staticmethod
    def available_grad_modes():
        return Registry.grad_modes

    @staticmethod
    def available_attacks():
        return Registry.all_attacks

    @staticmethod
    def dataset(dataset_name, **kwargs):
        Registry.__check_dataset(dataset_name)
        return Registry.datamap[dataset_name](**kwargs)

    @staticmethod
    def dm_class(dataset_name, diffusion_type="vpsde"):
        Registry.__check_dataset(dataset_name)
        assert isinstance(diffusion_type, str)
        if not diffusion_type in Registry.available_dbp_schemes():
            logger.error(f"{diffusion_type} is not a supported DBP scheme")
            exit(1)

        if "vp" in diffusion_type and dataset_name == "cifar10":
            dataset_name = "cifar10_vp"
        elif dataset_name == "cifar10":
            dataset_name = "cifar10_ddpm"
        return Registry.dm_map[dataset_name]

    @staticmethod
    def classifier(dataset_name, classifier_name=None, **kwargs):
        Registry.__check_dataset(dataset_name)
        if classifier_name is None:
            classifier_name = Registry.default_classifiers[dataset_name]
        assert isinstance(classifier_name, str)
        if classifier_name not in Registry.available_classifiers(dataset_name):
            logger.error(
                f"{classifier_name} is not an available classifiers for {dataset_name}"
            )
            exit(1)

        if dataset_name == "youtube" or (
            dataset_name == "cifar10" and classifier_name == "VGG16"
        ):
            from .classifiers.tf import RESNET50NP, VGG16

        return eval(classifier_name)(**kwargs)

    @staticmethod
    def default_loss(attack_name):
        assert isinstance(attack_name, str)
        if attack_name not in Registry.available_attacks():
            logger.error(f"requested attack not available")
            exit(1)
        if attack_name in ["LF", "diffattack_LF", "diffattack_apgd", "apgd", "stadv"]:
            return MarginLoss(kappa="inf")
        elif attack_name in ["ppgd", "lagrange"]:
            return MarginLoss(kappa=1)
        elif attack_name == "pgd":
            return CE()
        else:
            return None

    @staticmethod
    def attack_params(dataset_name, attack_name):
        Registry.__check_dataset(dataset_name)

        assert isinstance(attack_name, str)
        if attack_name not in Registry.available_attacks():
            logger.error(f"requested attack not available")
            exit(1)

        oname = attack_name
        attack_name = attack_name.replace("diffattack_", "")

        params = eval(attack_name + "_params")
        params["attack_name"] = oname
        params["eot_iters"] = 1 if dataset_name == "cifar10" else 2

        if attack_name in ["ppgd", "lagrange"]:
            params["lpips_model"] = (
                "alexnet" if dataset_name != "cifar10" else "alexnet_cifar"
            )

        if attack_name in ["pgd", "apgd"]:
            if dataset_name in ["celeba-hq", "youtube"]:
                params["eps"] = 0.0627
            elif dataset_name == "cifar10":
                params["eps"] = 0.0314
            else:
                params["eps"] = 0.0156

        if attack_name == "LF":
            params["initial_const"] = 1.0e8 if dataset_name == "cifar10" else 1.0e4
            if dataset_name == "cifar10":
                params["filter_args"]["kernels"] = [[5, 5], [7, 7], [5, 5], [3, 3]]
            else:
                params["filter_args"]["kernels"] = [
                    [21, 5],
                    [5, 5],
                    [17, 33],
                    [7, 7],
                    [47, 5],
                    [33, 17],
                    [17, 17],
                    [5, 5],
                    [3, 3],
                ]

        return params

    @staticmethod
    def default_dbp_params(diffusion_type="vpsde", grad_mode="full", grad_powers=None):
        assert isinstance(diffusion_type, str)
        if not diffusion_type in Registry.available_dbp_schemes():
            logger.error(f"{diffusion_type} is not a supported DBP scheme")
            exit(1)
        if not grad_mode in Registry.available_grad_modes():
            logger.error(f"{grad_mode} is not a supported gradient mode")
            exit(1)

        assert (
            isinstance(grad_powers, tuple)
            or isinstance(grad_powers, list)
            or grad_powers is None
        )
        if grad_powers is not None:
            assert len(grad_powers) == 2
            assert isinstance(grad_powers[0], float) or isinstance(grad_powers[0], int)
            assert isinstance(grad_powers[1], float) or isinstance(grad_powers[1], int)

        return {
            "beta_min": 0.1,
            "beta_max": 20.0,
            "full_horizon_length": 1000,
            "diffusion_t_mode": "same",
            "diffusion_type": diffusion_type,
            "diffusion_repeats": 1,
            "timestep_respacing": None,
            "guidance_mode": None,
            "guidance_scale": 60000.0,  # 70000.0 for SSIM and cifar10
            "guidance_ptb": 8.0,
            "deterministic": False,
            "grad_mode": grad_mode,
            "grad_powers": grad_powers,
        }

    def dbp_params(
        dataset_name, diffusion_type="vpsde", grad_mode="full", grad_powers=None
    ):
        Registry.__check_dataset(dataset_name)
        params = Registry.default_dbp_params(
            diffusion_type=diffusion_type, grad_mode=grad_mode, grad_powers=grad_powers
        )
        params["batch_size"] = 8

        if dataset_name == "cifar10":
            params["batch_size"] = 128
            params["diffusion_steps"] = 100 if "vp" in diffusion_type else 36
            if "ddpm" in diffusion_type:
                params["diffusion_t_mode"] = "plus1"
                params["diffusion_repeats"] = 4
                params["guidance_mode"] = "MSE"
        elif dataset_name == "imagenet":
            params["diffusion_steps"] = 150 if "vp" in diffusion_type else 45
            if "ddpm" in diffusion_type:
                params["timestep_respacing"] = 250
                params["guidance_mode"] = "SSIM"
                params["guidance_scale"] = 1000.0
                params["guidance_ptb"] = 4.0
        else:
            params["diffusion_steps"] = 500 if dataset_name == "celeba-hq" else 200
            if "ddpm" in diffusion_type:
                params["guidance_mode"] = "SSIM"
                params["guidance_scale"] = 1000.0
                params["guidance_ptb"] = 4.0

        return params
