import os
import logging
import shutil
import argparse
import random
import numpy as np
import yaml
import torch
from importlib import resources
from PIL import Image, ImageFont, ImageDraw, ImageOps
from torchvision import transforms

import DiffBreak.resources.assets as assets

from ..attacks import ID
from .initializer import Initializer
from .classifiers.torch.classifier import Classifier
from .logs import get_logger

logger = get_logger()
torch.set_float32_matmul_precision("high")

DEFAULT_EXP_CONF = {
    "excluded": [],
    "resume": 0,
    "total_success": 0,
    "total_single_success": 0,
}


def get_conf(out_dir):
    exp_conf = {k: v for k, v in DEFAULT_EXP_CONF.items()}
    exp_conf["results_file"] = os.path.join(out_dir, "results.txt")
    exp_conf["excluded_db"] = os.path.join(out_dir, "excluded.yaml")
    exp_conf["out_dir"] = out_dir
    return exp_conf


def id_attack(attack):
    return isinstance(attack, ID)


class Runner:
    def __init__(
        self,
        exp_conf,
        dataset,
        classifier,
        loss,
        dm_class=None,
    ):
        total_samples, balanced_splits = (
            exp_conf["params"]["data"]["total_samples"],
            exp_conf["params"]["data"]["balanced_splits"],
        )
        dbp_params = exp_conf["params"].get("dbp", {}).get("params", None)
        targeted, attack_params = (
            exp_conf["params"]["attack"]["targeted"],
            exp_conf["params"]["attack"]["params"],
        )
        eval_mode, verbose = (
            exp_conf["params"]["eval"]["eval_mode"],
            exp_conf["params"]["eval"]["verbose"],
        )
        seed, save_image_mode = (
            exp_conf["params"]["exp"]["seed"],
            exp_conf["params"]["exp"]["save_image_mode"],
        )

        data = Initializer.data(
            dataset, total_samples=total_samples, balanced_splits=balanced_splits
        )

        if dm_class is not None and dbp_params is not None:
            dbp = Initializer.dbp(dm_class, data.image_size, **dbp_params)
        else:
            dbp = None
        defended_model = Initializer.defended_classifier(
            classifier, loss_fn=loss, eval_mode=eval_mode, verbose=verbose, dbp=dbp
        )
        self.attack = Initializer.attack(
            defended_model, targeted=targeted, **attack_params
        )
        assert isinstance(seed, int)
        self.set_seed(self.attack.model, seed=seed)

        assert save_image_mode in ["none", "successful", "originally_failed"]
        assert (
            isinstance(exp_conf["resume"], int) or isinstance(exp_conf["resume"], float)
        ) and exp_conf["resume"] >= 0
        assert (
            isinstance(exp_conf["total_success"], int)
            or isinstance(exp_conf["total_success"], float)
        ) and exp_conf["total_success"] >= 0
        assert (
            isinstance(exp_conf["total_single_success"], int)
            or isinstance(exp_conf["total_single_success"], float)
        ) and exp_conf["total_single_success"] >= 0
        assert isinstance(exp_conf["results_file"], str)
        assert isinstance(exp_conf["excluded_db"], str)

        self.out_dir = exp_conf["out_dir"]
        self.excluded = exp_conf["excluded"]
        self.resume = exp_conf["resume"]
        self.total_success = exp_conf["total_success"]
        self.total_single_success = exp_conf["total_single_success"]
        self.results_file = exp_conf["results_file"]
        self.excluded_db = exp_conf["excluded_db"]
        self.do_save = save_image_mode
        self.num_samples = data.num_total_samples
        self.data = iter(data)
        self.start = 0
        self.global_ctr = -1
        self.device = "cpu"

    @staticmethod
    def __load_state(exp_conf, attack_name, eval_mode):
        exp_conf = {k: v for k, v in exp_conf.items()}
        results_file = exp_conf["results_file"]
        assert os.path.exists(results_file)
        excluded_db = exp_conf["excluded_db"]
        with open(results_file) as f:
            lines = f.readlines()

        if attack_name != "id":
            results_single = [float(line.strip().split(" ")[-1]) for line in lines]
            results = (
                [float(line.strip().split(" ")[-3][:-1]) for line in lines]
                if eval_mode != "single"
                else [r for r in results_single]
            )
            results_single = np.array(results_single)
            results = np.array(results)
            exp_conf["total_success"] = results.sum()
            exp_conf["total_single_success"] = results_single.sum()
        else:
            exp_conf["total_success"] = 0.0
            exp_conf["total_single_success"] = 0.0

        exp_conf["resume"] = len(lines)
        if os.path.exists(excluded_db):
            with open(excluded_db) as excluded_db_handle:
                exp_conf["excluded"] = yaml.load(
                    excluded_db_handle, Loader=yaml.Loader
                )["idx"]
        return exp_conf

    @staticmethod
    def resume(out_dir):
        param_file = os.path.join(out_dir, "config.yaml")
        assert os.path.exists(param_file)
        with open(param_file) as f:
            params = yaml.load(f, Loader=yaml.Loader)
        eval_mode = params["eval"]["eval_mode"]
        attack_name = params["attack"]["params"]["attack_name"]
        exp_conf = Runner.__load_state(get_conf(out_dir), attack_name, eval_mode)
        exp_conf["params"] = params
        return exp_conf

    @staticmethod
    def __parse_params(
        attack_params,
        dbp_params,
        targeted,
        eval_mode,
        total_samples,
        balanced_splits,
        verbose,
        seed,
        save_image_mode,
    ):
        return {
            "data": {
                "total_samples": total_samples,
                "balanced_splits": balanced_splits,
            },
            "dbp": {
                "params": dbp_params,
            },
            "attack": {
                "targeted": targeted,
                "params": attack_params,
            },
            "eval": {
                "eval_mode": eval_mode,
                "verbose": verbose,
            },
            "exp": {
                "seed": seed,
                "save_image_mode": save_image_mode,
            },
        }

    @staticmethod
    def setup(
        out_dir,
        attack_params,
        dbp_params=None,
        targeted=False,
        eval_mode="batch",
        total_samples=256,
        balanced_splits=False,
        verbose=2,
        seed=1234,
        save_image_mode="originally_failed",
        overwrite=False,
    ):
        assert isinstance(out_dir, str)
        assert attack_params is not None and isinstance(attack_params, dict)
        results_file = os.path.join(out_dir, "results.txt")
        param_file = os.path.join(out_dir, "config.yaml")
        if os.path.exists(out_dir):
            if not overwrite:
                logger.error(
                    "output directory exists! Provide overwrite=True to remove it,"
                    " or use method resume to continue instead"
                )
                exit(1)
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        exp_conf = get_conf(out_dir)

        cfg = Runner.__parse_params(
            attack_params=attack_params,
            dbp_params=dbp_params,
            targeted=targeted,
            eval_mode=eval_mode,
            total_samples=total_samples,
            balanced_splits=balanced_splits,
            verbose=verbose,
            seed=seed,
            save_image_mode=save_image_mode,
        )

        assert cfg["exp"]["save_image_mode"] in [
            "none",
            "successful",
            "originally_failed",
        ]
        if cfg["exp"]["save_image_mode"] != "none":
            os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
        with open(param_file, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        exp_conf["params"] = cfg
        return exp_conf

    def sample(self):
        self.global_ctr += 1
        x_init, y_init = next(self.data)
        x_init, y_init = x_init.to(self.device), y_init.to(self.device)
        orig_label = int(torch.argmax(y_init, dim=-1).item())
        if self.attack.targeted:
            target = torch.tensor(
                random.sample(
                    list(set(list(range(y_init.shape[-1]))) - set([orig_label])), 1
                ),
                device=x_init.device,
            )
            target_label = target.detach().cpu().item()
        else:
            target = torch.argmax(y_init, dim=-1)
            target_label = orig_label

        return x_init, y_init, target, target_label, orig_label

    def set_seed(self, model, seed=1234):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

        if not isinstance(model.model_fn, Classifier):
            import tensorflow as tf

            tf.random.set_seed(seed)
            tf.experimental.numpy.random.seed(seed)
            tf.random.set_seed(seed)
            os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
            os.environ["TF_DETERMINISTIC_OPS"] = "1"

    def save_imgs(
        self,
        orig_imgs,
        adv_imgs,
    ):
        start = int(self.start)
        output_dir = os.path.join(self.out_dir, "images")
        image_size = orig_imgs.shape[-1]
        topil = lambda x: transforms.ToPILImage()(x.squeeze().detach().cpu())
        orig_imgs = [topil(img) for img in orig_imgs]
        adv_imgs = [topil(img) for img in adv_imgs]

        for i, images in enumerate(zip(*[orig_imgs, adv_imgs])):
            new_im = Image.new("RGB", (image_size * 2, image_size))
            x_offset, draw_offsets = 0, [0]
            for im in images:
                new_im.paste(im, (x_offset, 0))
                x_offset += image_size
                draw_offsets.append(x_offset)
                x = transforms.ToTensor()(im)
            draw_offsets = draw_offsets[:-1]
            new_im = ImageOps.expand(new_im, border=20, fill=(255, 255, 255))
            draw = ImageDraw.Draw(new_im)

            with resources.path(assets, "arial.ttf") as fspath:
                font_path = fspath.as_posix()

            font = ImageFont.truetype(font_path, 15)
            for name, offset in zip(["orig", "adv"], draw_offsets):
                draw.text(
                    (offset + int(image_size / 2), 0),
                    name,
                    (0, 0, 0),
                    font=font,
                )
            new_im.save(os.path.join(output_dir, "img_" + str(i + start) + ".png"))

    def update_excluded(self):
        with open(self.excluded_db, "w", encoding="utf-8") as yaml_file:
            dump = yaml.dump(
                {"idx": self.excluded},
                default_flow_style=False,
                allow_unicode=True,
                encoding=None,
            )
            yaml_file.write(dump)

    def eval_init(self, x_init, y_init):
        init_success_rate, init_score = self.attack.model.eval_sample(
            x_init,
            y_init,
        )
        single_succ_cond = init_success_rate == 1.0
        succ_cond = (
            float(init_score) == 1.0
            if self.attack.model.eval_mode == "batch"
            else single_succ_cond
        )
        return succ_cond, init_success_rate, single_succ_cond

    def do_sample(self, x_init, y_init, target, succ_cond):
        x_adv, result, result_single = (
            self.attack(x_init, target, y_init)
            if (
                not id_attack(self.attack)
                and (self.attack.targeted or float(succ_cond) == 1.0)
            )
            else (x_init, 1.0, 1.0)
        )
        return x_adv, float(result), float(result_single)

    def update_stats(
        self,
        x_init,
        x_adv,
        orig_label,
        target,
        succ_cond,
        single_succ_cond,
        result,
        result_single,
    ):
        self.total_single_success += result_single
        self.total_success += result
        portion_success = self.total_success / self.start
        portion_success_single = self.total_single_success / self.start

        if self.do_save == "originally_failed":
            should_save = (result == 1 and float(succ_cond) == 1) or (
                result_single == 1 and float(single_succ_cond) == 1
            )
        elif self.do_save == "successful":
            should_save = result == 1 or result_single == 1
        else:
            should_save = False
        if should_save:
            self.save_imgs(
                x_init.detach(),
                x_adv.detach(),
            )
        message = f"original label: {orig_label}"
        if self.attack.targeted:
            message = message + f", target label: {target.detach().cpu().item()}"
        message = message + f", originally robust: {float(succ_cond)}"
        if not id_attack(self.attack):
            message = message + f", result: {result}"
            if self.attack.model.eval_mode != "single":
                message = message + f", result-single: {result_single}"
        with open(self.results_file, "a") as f:
            f.write(message + "\n")

        if not id_attack(self.attack):
            message = f"total success rate: {portion_success}"
            if self.attack.model.eval_mode != "single":
                message = message + f", single success rate: {portion_success_single}"
            logger.info(message + "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

    def to(self, device):
        self.device = device
        self.attack = self.attack.to(device)
        return self

    def eval(self):
        self.attack = self.attack.eval()
        return self

    def execute(self):
        while self.start < self.num_samples:
            x_init, y_init, target, target_label, orig_label = self.sample()
            if self.global_ctr in self.excluded:
                continue

            if self.start < self.resume:
                self.start += 1
                continue

            logger.bar("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            message = f"sample: {self.start}, orig label: {orig_label}"
            if self.attack.targeted:
                message = message + f"target label: {target_label}"
            logger.info(message)

            succ_cond, init_success_rate, single_succ_cond = self.eval_init(
                x_init, y_init
            )

            if self.attack.targeted and init_success_rate != 1:
                if self.global_ctr not in self.excluded:
                    self.excluded.append(self.global_ctr)
                    self.update_excluded()
                logger.info(
                    "Mode is targeted but the sample is not originally classified correctly."
                    " Skipping this one."
                )
                continue
            self.start += 1

            x_adv, result, result_single = self.do_sample(
                x_init, y_init, target, succ_cond
            )
            self.update_stats(
                x_init,
                x_adv,
                orig_label,
                target,
                succ_cond,
                single_succ_cond,
                result,
                result_single,
            )
