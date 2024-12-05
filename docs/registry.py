import logging
import sys
from importlib import resources

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

import docs.snippets as resource_module

WIDTH = 100

cifar10_description_map = {
    "WIDERESNET_28_10": "The WideResNet-28-10 from [bold]RobustBench[/bold] [purple][underline]https://github.com/RobustBench/robustbench[/underline][/purple]",
    "WRN_28_10_AT0": "Adversarially trained WRN-70-16 classifier by [bold]Gowal et al.[/bold] [purple][underline]https://arxiv.org/abs/2110.09468[/underline][/purple] "
    '\n [white]•[/] Taken from RobustBench (threat_model="Linf", model name="Gowal2021Improving_28_10_ddpm_100m")',
    "WRN_28_10_AT1": "Adversarially trained WRN-70-16 classifier by [bold]Gowal et al.[/bold] [purple][underline]https://arxiv.org/abs/2010.03593[/underline][/purple] "
    '\n [white]•[/] Taken from RobustBench (threat_model="Linf", model name="Gowal2020Uncovering_28_10_extra")',
    "WRN_70_16_AT0": "Adversarially trained WRN-70-16 classifier by [bold]Gowal et al.[/bold] [purple][underline]https://arxiv.org/abs/2110.09468[/underline][/purple] "
    '\n [white]•[/] Taken from RobustBench (threat_model="Linf", model name="Gowal2021Improving_70_16_ddpm_100m")',
    "WRN_70_16_AT1": "Adversarially trained WRN-70-16 classifier by [bold]Rebuffi et al.[/bold] [purple][underline]https://arxiv.org/pdf/2103.01946[/underline][/purple] "
    '\n [white]•[/] Taken from RobustBench (threat_model="Linf", model name="Rebuffi2021Fixing_70_16_cutmix_extra")',
    "WRN_70_16_L2_AT1": "Adversarially trained WRN-70-16 classifier by [bold]Rebuffi et al.[/bold] [purple][underline]https://arxiv.org/pdf/2103.01946[/underline][/purple] "
    '\n [white]•[/] Taken from RobustBench (threat_model="L2", model name="Rebuffi2021Fixing_70_16_cutmix_extra").',
    "WIDERESNET_70_16": "The WideResNet-70-16 classifier by [bold]DiffPure[/bold] [purple][underline]https://arxiv.org/pdf/2205.07460[/underline][/purple] "
    "(default).",
    "RESNET_50": "The ResNet50 classifier by [bold]DiffPure[/bold] [purple][underline]https://arxiv.org/pdf/2205.07460[/underline][/purple].",
    "WRN_70_16_DROPOUT": "The wrn-70-16-dropout by [bold]DiffPure[/bold] [purple][underline]https://arxiv.org/pdf/2205.07460[/underline][/purple].",
    "VGG16": "A pretrained Tensorflow (TF) [bold]VGG16[/bold] classifier.",
}
imagenet_description_map = {
    "RESNET18": "ResNet18 from [bold]torchvision.models[/bold].",
    "RESNET50": "ResNet50 from [bold]torchvision.models[/bold].",
    "RESNET101": "ResNet101 from [bold]torchvision.models[/bold].",
    "WIDERESNET_50_2": "WideResNet-50-2 from [bold]torchvision.models[/bold].",
    "DEIT_S": "ViT classifier by [bold]facebookresearch[/bold] (default).",
}
celeba_description_map = {
    "NET_BEST": "The attribute classifiers "
    "from [purple][underline]https://github.com/chail/gan-ensembling[/purple][/underline] "
    "(default)."
}
youtube_description_map = {
    "RESNET50NP": "A pretrained Tensorflow (TF) ResNet50 classifier " "(default)."
}

celeb_atts = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]

dataset_description_map = {
    "cifar10": "The known Cifar10 dataset: [purple][underline]https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf[/underline][/purple].",
    "imagenet": "The known ImageNet dataset [purple][underline]https://ieeexplore.ieee.org/document/5206848[/underline][/purple].",
    "celeba-hq": "The CelebA-Hq dataset [purple][underline]https://arxiv.org/abs/1710.10196[/underline][/purple] containing faces "
    "of celebrities for multiple binary attribute classification tasks.",
    "youtube": "The YouTube-faces datasets [purple][underline]https://ieeexplore.ieee.org/document/5995566[/underline][/purple] "
    "wherein the task is to classify 1283 faces correctly.",
}

classifier_description_map = {
    "cifar10": cifar10_description_map,
    "imagenet": imagenet_description_map,
    "celeba-hq": celeba_description_map,
    "youtube": youtube_description_map,
}

dm_description_map = {
    "GuidedModel": "The commonly used diffusion model for [cyan]imagenet[/cyan] by "
    "[bold]Dhariwal & Nichol[/bold] [purple][underline]https://arxiv.org/pdf/2105.05233[/underline][/purple].",
    "DDPMModel": "The [cyan]celeba-hq[/cyan] pretrained diffusion model from the [bold]SDEdit[/bold] "
    "library [purple][underline]https://github.com/ermongroup/SDEdit[/underline][/purple].",
    "ScoreSDEModel": "The Score SDE diffusion model for [cyan]cifar10[/cyan] by [bold]Song et al.[/bold] "
    "[purple][underline]https://arxiv.org/pdf/2011.13456[/underline][/purple]. Default for VP-based.",
    "HaoDDPM": "The diffusion model for [cyan]cifar10[/cyan] by [bold]Ho et al.[/bold] "
    "[purple][underline]https://arxiv.org/pdf/2006.11239[/underline][/purple]. Default for ddpm.",
}

grad_mode_description_map = {
    "full": "The full accurate gradient computed efficiently using our [bold]DiffGrad[/bold] "
    "module [purple][underline]https://arxiv.org/pdf/2411.16598[/underline][/purple].",
    "full_intermediate": "Similar to [cyan]full[/cyan] but additionally computes the loss "
    "function for each and every step of the reverse pass and "
    "adds its gradients to the backpropagated total.",
    "adjoint": 'The known "adjoint" [purple][underline]https://arxiv.org/pdf/2001.01328[/underline][/purple] method '
    "for VP-based BDP (cannot be used with ddpm schemes). "
    "DiffBreak fixes the implementation issues of [bold]torchsde[/bold] "
    "[purple][underline]https://github.com/google-research/torchsde[/underline][/purple] and provides "
    "a far more powerful tool as a result.",
    "bpda": "Backward-pass Differentiable Approximation [purple][underline]https://arxiv.org/pdf/1802.00420[/underline][/purple] "
    "(DBP is replaced by the identity function during backprop.)",
    "blind": "Gradients are obtained by attacking the classifier directly and without involving "
    "DBP at all. The defense is only considered when the attack sample is evaluated.",
    "forward_diff_only": "Similar to [cyan]blind[/cyan] but instead adds the full noise from the forward "
    "pass of DBP to the sample and uses this noisy output to invoke the "
    "classifier and obtain the gradients.",
}

scheme_description_map = {
    "vpsde": "The VP-SDE-Based DBP scheme (i.e., [bold]DiffPure[/bold] [purple][underline]https://arxiv.org/pdf/2205.07460[/underline][/purple]).",
    "ddpm": "The DDPM-based DBP scheme ([purple][underline]https://arxiv.org/pdf/2205.14969[/underline][/purple]).",
    "vpode": 'Similar to "vpsde" but performs purification by solving an ODE in the reverse pass '
    "instead of an SDE. This method was also proposed in DiffPure [purple][underline]https://arxiv.org/pdf/2205.07460[/underline][/purple].",
    "ddpm_ode": "Similar to [cyan]vpode[/cyan] but implemented for discrete-time DBP (i.e., DDPM).",
}

attack_description_map = {
    "pgd": "The [bold]PGD[/bold] attack: [purple][underline]https://arxiv.org/pdf/1607.02533[/underline][/purple].",
    "ppgd": "[bold]PerceptualPGDAttack[/bold]: [purple][underline]https://arxiv.org/abs/2006.12655[/underline][/purple].",
    "apgd": "[bold]AutoAttack[/bold] (Linf): [purple][underline]https://arxiv.org/pdf/2003.01690[/underline][/purple].",
    "diffattack_apgd": "[bold]DiffAttack[/bold]: [purple][underline]https://arxiv.org/pdf/2311.16124[/underline][/purple].",
    "LF": "Our [bold]Low-Frequency[/bold] attack: [purple][underline]https://arxiv.org/pdf/2411.16598[/underline][/purple].",
    "diffattack_LF": "Our [cyan]LF[/cyan] attack augmented with the per-step losses used by [bold]DiffAttack[/bold].",
    "stadv": "The [bold]StAdv[/bold] attack: [purple][underline]https://arxiv.org/pdf/1801.02612[/underline][/purple].",
    "lagrange": "[bold]LagrangePerceptualAttack[/bold]: [purple][underline]https://arxiv.org/abs/2006.12655[/underline][/purple].",
    "id": "No attack. Use this to evaluate clean accuracy.",
}

command_desc_map = {
    "help": "Prints this message.",
    "datasets": "Lists all available datasets our registry",
    "classifiers [DATASET_NAME]": "Lists all classifiers available in our registry for DATASET_NAME. If DATASET_NAME "
    "is not provided, lists all classifiers for all datasets.",
    "dm_classes DATASET_NAME": "Lists all diffusion models available in our registry for DATASET_NAME. If DATASET_NAME "
    "is not provided, lists all diffusion models for all datasets.",
    "dbp_schemes": "Lists all available DBP schemes suppotred by DiffBreak.",
    "attacks": "Lists all attacks offered by DiffBreak.",
    "losses": "Lists all loss functions you can use in your attacks.",
    "grad_modes": "Lists all gradient modes available in DiffBreak.",
    "custom classifier|dm_class|dataset": "Provides an explanation regarding the use of a custom classifier, dm_class or dataset.",
}

loss_function_map = {
    "CE": "The cross-entropy loss. Default for pgd.",
    "MarginLoss": "The max-margin loss: [purple][underline]https://arxiv.org/pdf/1608.04644[/underline][/purple]. "
    "Default for all remaining attacks.",
    "DLR": "The Difference of Logits Ratio loss: [purple][underline]https://arxiv.org/pdf/2003.01690[/underline][/purple]",
}


class Registry:
    all_datasets = ["cifar10", "celeba-hq", "imagenet", "youtube"]
    dbp_schemes = ["vpsde", "vpode", "ddpm", "ddpm_ode"]

    commands = [
        "help",
        "datasets",
        "classifiers",
        "dm_classes",
        "dbp_schemes",
        "attacks",
        "losses",
        "grad_modes",
        "custom",
    ]

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

    all_losses = [
        "CE",
        "MarginLoss",
        "DLR",
    ]

    @staticmethod
    def available_losses():
        return Registry.all_losses

    @staticmethod
    def available_commands():
        return Registry.commands

    @staticmethod
    def available_datasets():
        return Registry.all_datasets

    def __check_dataset(dataset_name):
        assert isinstance(dataset_name, str)
        if not dataset_name in Registry.available_datasets():
            logging.error(
                f"The registry does not contain default datasets, classifiers, dms or attacks "
                f"for dataset {dataset_name}."
            )
            exit(1)

    @staticmethod
    def available_classifiers(dataset_name):
        Registry.__check_dataset(dataset_name)
        return Registry.all_classifiers[dataset_name]

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
    def available_dms_for_dataset(dataset):
        return [v for k, v in Registry.dm_names.items() if dataset in k]


def make_dataset_entry(dataset):
    description = dataset_description_map[dataset]
    return f"[cyan]-{dataset}[/]\n{description}\n"


def make_attack_entry(attack):
    description = attack_description_map[attack]
    return "[cyan]" + f"-{attack}[/]\n{description}\n"


def make_loss_entry(loss):
    description = loss_function_map[loss]
    return "[cyan]" + f"-{loss}[/]\n{description}\n"


def make_dbp_entry(dbp):
    description = scheme_description_map[dbp]
    return "[cyan]" + f"-{dbp}[/]\n{description}\n"


def make_grad_entry(grad):
    description = grad_mode_description_map[grad]
    return "[cyan]" + f"-{grad}[/]\n{description}\n"


def make_classifier_entry(classifier, dataset):
    description = classifier_description_map[dataset][classifier]
    return "[cyan]" + f"-{classifier}[/]\n{description}\n"


def make_dm_entry(dm):
    description = dm_description_map[dm]
    return "[cyan]" + f"-{dm}[/]\n{description}\n"


def make_record_table():
    return Table(
        box=None,
        expand=False,
        show_header=False,
        show_edge=False,
        pad_edge=False,
        width=WIDTH,
    )


def make_command_table():
    command_table = make_record_table()

    commands = Registry.available_commands()
    command_info = ""
    for command in commands:
        key = [k for k in command_desc_map.keys() if command in k][0]
        desc = command_desc_map[key]
        command_info = command_info + "[cyan]" + f"-{key}[/]\n{desc}\n"
    command_table.add_row(command_info)
    return command_table


def make_data_table():
    data_table = make_record_table()

    data_info = ""
    for dataset in Registry.available_datasets():
        data_info = data_info + make_dataset_entry(dataset)
        if dataset == "celeba-hq":
            data_info = (
                data_info + " [white]•[/] [bold italic]Available attributes:[/]\n"
            )
            for att in celeb_atts:
                data_info = data_info + f"[blue]   -- {att}[/blue]\n"

    data_table.add_row(data_info)
    return data_table


def make_attack_table():
    attack_table = make_record_table()
    attack_info = ""
    for attack in Registry.available_attacks():
        attack_info = attack_info + make_attack_entry(attack)
    attack_table.add_row(attack_info)
    return attack_table


def make_loss_table():
    loss_info = ""
    loss_table = make_record_table()
    for loss in Registry.available_losses():
        loss_info = loss_info + make_loss_entry(loss)
    loss_table.add_row(loss_info)
    return loss_table


def make_dbp_table():
    dbp_info = ""
    dbp_table = Table(
        box=None,
        expand=False,
        show_header=False,
        show_edge=False,
        pad_edge=False,
        width=120,
    )
    for dbp in Registry.available_dbp_schemes():
        dbp_info = dbp_info + make_dbp_entry(dbp)
    dbp_table.add_row(dbp_info)
    return dbp_table


def make_grad_table():
    grad_info = ""
    grad_table = make_record_table()
    for grad in Registry.available_grad_modes():
        grad_info = grad_info + make_grad_entry(grad)

    grad_table.add_row(grad_info)
    return grad_table


def make_classifier_table(dataset):
    classifier_info = ""
    classifier_table = make_record_table()
    for classifier in Registry.available_classifiers(dataset):
        classifier_info = classifier_info + make_classifier_entry(classifier, dataset)
    if dataset == "celeba-hq":
        classifier_info = (
            classifier_info + " [white]•[/] [bold italic]Available attributes:[/]\n"
        )
        for att in celeb_atts:
            classifier_info = classifier_info + f"[blue] -- {att}[/blue]\n"
    classifier_table.add_row(classifier_info)
    return classifier_table


def make_dm_table(dataset):
    dm_info = ""
    dm_table = make_record_table()
    for dm in Registry.available_dms_for_dataset(dataset):
        dm_info = dm_info + make_dm_entry(dm)
    dm_table.add_row(dm_info)
    return dm_table


def make_code_table(code):
    renderable1 = Syntax(code, "python3", line_numbers=True, indent_guides=True)
    code_table = Table(show_header=False, pad_edge=False, box=None, expand=True)
    code_table.add_column("1", ratio=0.5)
    code_table.add_row(renderable1)
    return code_table


def make_base_table():
    table = Table.grid(padding=0, pad_edge=False)
    table = Table(
        box=None,
        expand=False,
        show_header=False,
        show_edge=False,
        pad_edge=False,
    )
    table.title = "DiffBreak"
    return table


def make_command_cards():
    command_table = make_command_table()
    return ["[bold green]Available commands"], [command_table]


def make_data_cards():
    data_table = make_data_table()
    return ["[bold green]Available datasets"], [data_table]


def make_attack_cards():
    attack_table = make_attack_table()
    return ["[bold green]Available attacks"], [attack_table]


def make_loss_cards():
    loss_table = make_loss_table()
    return ["[bold green]Available losses"], [loss_table]


def make_dbp_cards():
    dbp_table = make_dbp_table()
    return ["[bold green]Available DBP schemes"], [dbp_table]


def make_grad_cards():
    grad_table = make_grad_table()
    return ["[bold green]Available gradient modes"], [grad_table]


def make_classifier_cards(dataset=None):
    if dataset is not None:
        if dataset not in Registry.available_datasets():
            logging.error("The provided dataset is not in the registry.")
            exit(1)
        datasets = [dataset]
    else:
        datasets = Registry.available_datasets()
    headers, tables = [], []
    for dataset in datasets:
        headers.append(f"[bold green]Available classifiers for {dataset}")
        tables.append(make_classifier_table(dataset))
    return headers, tables


def make_dm_cards(dataset=None):
    if dataset is not None:
        if dataset not in Registry.available_datasets():
            logging.error("The provided dataset is not in the registry.")
            exit(1)
        datasets = [dataset]
    else:
        datasets = Registry.available_datasets()
    headers, tables = [], []
    for dataset in datasets:
        headers.append(f"[bold green]Default diffusion models for {dataset}")
        tables.append(make_dm_table(dataset))
    return headers, tables


def make_code_card(message, code_path):
    with open(code_path) as f:
        code = f.read()
    code_table = make_code_table(code)
    return [message], [code_table]


def make_dm_subclass_card():
    resource_file = "dm_subclass.txt"
    message = "[bold green]Providing a custom score model\n"
    with resources.path(resource_module, resource_file) as fspath:
        resource_path = fspath.as_posix()
    return make_code_card(message, resource_path)


def make_data_subclass_card():
    resource_file = "data_subclass.txt"
    message = "[bold green]Providing a custom dataset\n"
    with resources.path(resource_module, resource_file) as fspath:
        resource_path = fspath.as_posix()
    return make_code_card(message, resource_path)


def make_custom_classifier_card():
    resource_file = "classifier.txt"
    message = "[bold green]Providing a custom pretrained classifier\n"
    with resources.path(resource_module, resource_file) as fspath:
        resource_path = fspath.as_posix()
    return make_code_card(message, resource_path)


def make_custom_cards(ctype):
    if ctype not in ["dataset", "dm_class", "classifier"]:
        logging.error(
            "Custom implementations are only available for dataset|classifier|dm_class."
        )
        exit(1)

    if ctype == "dataset":
        return make_data_subclass_card()
    if ctype == "dm_class":
        return make_dm_subclass_card()
    return make_custom_classifier_card()


def make_response(command, card_fn, card_args):
    console = Console(
        force_terminal=True,
    )

    table = make_base_table()
    table.title = f"[bold red]{table.title} {command}[/]"
    headers, cards = card_fn(*card_args)
    for header, card in zip(headers, cards):
        table.add_row(header)
        table.add_row(card)
    c = Console(record=True)
    c.print(table)


def registry():
    if len(sys.argv) == 1:
        make_response("help", make_command_cards, [])
        exit(0)

    command = sys.argv[1]

    if command not in Registry.available_commands():
        make_response("help", make_command_cards, [])
        exit(1)

    args = sys.argv[2:]

    if command not in ["classifiers", "dm_classes", "custom"]:
        if len(args) > 0:
            make_response("help", make_command_cards, [])
            exit(1)
    elif len(args) > 1:
        make_response("help", make_command_cards, [])
        exit(1)

    if command == "help":
        card_fn = make_command_cards
    elif command == "datasets":
        card_fn = make_data_cards
    elif command == "losses":
        card_fn = make_loss_cards
    elif command == "dbp_schemes":
        card_fn = make_dbp_cards
    elif command == "grad_modes":
        card_fn = make_grad_cards
    elif command == "classifiers":
        card_fn = make_classifier_cards
    elif command == "attacks":
        card_fn = make_attack_cards
    elif command == "dm_classes":
        card_fn = make_dm_cards
    else:
        card_fn = make_custom_cards

    make_response(command, card_fn, args)


if __name__ == "__main__":
    registry()
