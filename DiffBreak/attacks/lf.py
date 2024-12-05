import numpy as np
import torch
from tqdm import trange
from torch.optim import Adam
from pytorch_forecasting.utils import unsqueeze_like
import kornia

from .utils import *

INF = float("inf")


class SGNOpt(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1):
        super().__init__(params, defaults={"lr": lr})

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                p.data -= group["lr"] * torch.sgn(p.grad.data)


class Filter(torch.nn.Module):
    def __init__(
        self,
        kernels,
        shape,
        box=(1, 1),
        sigma_color=0.1,
        norm=1,
        pad_mode="reflect",
        filter_mode=False,
        loss_factor=1,
        loss_norm=2,
    ):
        super().__init__()
        self.norm, self.sigma_color, self.pad_mode, self.box, self.filter_mode = (
            norm,
            sigma_color,
            pad_mode,
            box,
            filter_mode,
        )
        self.kernels = torch.nn.ParameterList(
            [torch.nn.Parameter(self.__get_init_w(kernel, shape)) for kernel in kernels]
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.loss_factor = loss_factor
        self.loss_norm = loss_norm

    def pad_w(self, w):
        return torch.nn.functional.pad(
            w, (0, w.shape[-1] - 1, 0, w.shape[-2] - 1), "reflect"
        )

    def __get_init_w(self, kernel, shape):
        repeats, _, h, w = shape
        box = self.box if self.box is not None else kernel
        boxes = [int(np.ceil(h / box[0])), int(np.ceil(w / box[1]))]
        num_boxes = boxes[0] * boxes[1]
        w = (
            kornia.filters.get_gaussian_kernel2d(kernel, torch.tensor([[0.2, 0.2]]))
            .unsqueeze(0)
            .repeat(repeats, num_boxes, 1, 1)
        )
        return (
            w[..., : int(kernel[0] // 2) + 1, : int(kernel[1] // 2) + 1]
            .clamp(1e-5, 0.999999)
            .log()
        )

    def get_dist(self, x, kernel, guidance=None, norm=None):
        norm = self.norm if norm is None else norm
        unf_inp = self.extract_patches(x, kernel)
        guidance = guidance if guidance is not None else x
        guidance = torch.nn.functional.pad(
            guidance,
            self._box_pad(guidance, kernel),
            mode=self.pad_mode,
        )

        return torch.pow(
            torch.norm(
                unf_inp
                - guidance.view(guidance.shape[0], guidance.shape[1], -1)
                .transpose(1, 2)
                .view(
                    guidance.shape[0],
                    unf_inp.shape[1],
                    unf_inp.shape[2],
                    guidance.shape[1],
                    1,
                ),
                p=norm,
                dim=-2,
                keepdim=True,
            ),
            2,
        )

    def __get_color_kernel(self, guidance, kernel):
        if self.sigma_color <= 0:
            return 1
        dist = self.get_dist(guidance.double(), kernel).float()
        ret = (
            (-0.5 / (self.sigma_color**2) * dist)
            .exp()
            .view(guidance.shape[0], dist.shape[1], dist.shape[2], -1, 1)
        )
        return torch.nan_to_num(ret, nan=0.0)

    def _box_pad(self, x, kernel):
        box = self.box if self.box is not None else kernel
        col = (
            box[1] - (x.shape[-1] - (x.shape[-1] // box[1]) * box[1]) % box[1]
        ) % box[1]
        row = (
            box[0] - (x.shape[-2] - (x.shape[-2] // box[0]) * box[0]) % box[0]
        ) % box[0]
        return [0, col, 0, row]

    def _kernel_pad(self, kernel):
        return [
            (kernel[1] - 1) // 2,
            (kernel[1] - 1) - (kernel[1] - 1) // 2,
            (kernel[0] - 1) // 2,
            (kernel[0] - 1) - (kernel[0] - 1) // 2,
        ]

    def _median_pad(self, x, kernel, stride):
        ph = (
            (kernel[0] - stride[0])
            if x.shape[-2] % stride[0] == 0
            else (kernel[0] - (x.shape[-2] % stride[0]))
        )
        pw = (
            (kernel[1] - stride[1])
            if x.shape[-1] % stride[1] == 0
            else (kernel[1] - (x.shape[-1] % stride[1]))
        )
        return (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)

    def _compute_median(self, x, kernel):
        stride = kernel if not self.filter_mode else (1, 1)
        x_p = torch.nn.functional.pad(
            x, self._median_pad(x, kernel, stride), mode="reflect"
        )
        x_unf = x_p.unfold(2, kernel[0], stride[0]).unfold(3, kernel[1], stride[1])
        median = x_unf.contiguous().view(x_unf.size()[:4] + (-1,)).median(dim=-1)[0]
        return (
            median.unsqueeze(-2)
            .unsqueeze(-1)
            .repeat(
                1,
                1,
                1,
                x_p.shape[-2] // median.shape[-2],
                1,
                x_p.shape[-1] // median.shape[-1],
            )
            .flatten(2, 3)
            .flatten(-2)[..., : x.shape[-2], : x.shape[-1]]
        )

    def extract_patches(self, x, kernel):
        box = self.box if self.box is not None else kernel
        kern = (box[0] + (kernel[0] - 1), box[1] + (kernel[1] - 1))
        pad = [
            b + k for b, k in zip(self._box_pad(x, kernel), self._kernel_pad(kernel))
        ]
        inp_unf = (
            torch.nn.functional.pad(x, pad, mode=self.pad_mode)
            .unfold(2, kern[0], box[0])
            .unfold(3, kern[1], box[1])
            .permute(0, 2, 3, 1, 4, 5)
            .flatten(-2)
            .reshape(-1, x.shape[1], kern[0], kern[1])
        )

        return (
            inp_unf.unfold(2, kernel[0], 1)
            .unfold(3, kernel[1], 1)
            .permute(0, 2, 3, 1, 4, 5)
            .flatten(-2)
            .reshape(
                x.shape[0],
                inp_unf.shape[0] // x.shape[0],
                -1,
                inp_unf.shape[1],
                kernel[0] * kernel[1],
            )
        )

    def __apply_filter(self, x, w, guidance=None):
        w = self.pad_w(w)
        kernel = (w.shape[-2], w.shape[-1])
        box = self.box if self.box is not None else kernel
        inp_unf = self.extract_patches(x, kernel)

        boxes = [
            int(np.ceil(x.shape[-2] / box[0])),
            int(np.ceil(x.shape[-1] / box[1])),
        ]
        color_kernel = self.__get_color_kernel(guidance, kernel)
        w = (
            self.softmax(w.view(w.shape[0], w.shape[1], -1))
            .unsqueeze(-2)
            .unsqueeze(-1)
            .repeat(1, 1, inp_unf.shape[2], 1, 1)
            * color_kernel
        ).view(w.shape[0], w.shape[1], inp_unf.shape[2], -1, 1)
        out = inp_unf.matmul(w).transpose(2, 3).squeeze(-1) / w.squeeze(-1).sum(
            -1
        ).unsqueeze(2)
        out = (
            out.view(-1, inp_unf.shape[-2], inp_unf.shape[-3])
            .reshape(x.shape[0], -1, x.shape[1] * box[0] * box[1])
            .transpose(2, 1)
        )
        return torch.nn.functional.fold(
            out,
            (boxes[0] * box[0], boxes[1] * box[1]),
            box,
            stride=box,
        )[..., : x.shape[-2], : x.shape[-1]]

    def __compute_filter_loss(self, x, guidance, kernel, norm=2):
        return self.get_dist(
            x, kernel, guidance=self._compute_median(guidance, kernel), norm=norm
        ).view(x.shape[0], -1).sum(-1, keepdims=True) / torch.prod(
            torch.tensor(x.shape[1:])
        )

    def compute_loss(self, x, guidance=None, normalize=False):
        guidance = x if guidance is None else guidance
        kernels = [(f.shape[-2] * 2 - 1, f.shape[-1] * 2 - 1) for f in self.kernels]
        if len(kernels) == 0 or self.loss_factor == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return torch.concat(
            [
                self.__compute_filter_loss(
                    x,
                    guidance,
                    k,
                    norm=self.loss_norm,
                )
                / (np.prod(k) if normalize else 1)
                for k in kernels
            ],
            dim=-1,
        ).sum(-1) * (
            np.array([np.prod(k) for k in kernels]).sum() if normalize else 1.0
        )

    def forward(self, x, guidance):
        for filt in self.kernels:
            x = self.__apply_filter(x, filt, guidance=guidance)
        return x.float()


class StepPerformer:
    def __init__(
        self,
        model_loss,
        cond_fn,
        eps,
        lr,
        optimizer_class,
        eot_iters=1,
        clip_min=0.0,
        clip_max=1.0,
        regularization_type=None,
        regularization_factor=0.0,
        regularization_threshold=None,
    ):
        assert regularization_type is None or regularization_type in ["l1", "l2"]
        if regularization_type is not None:
            assert (
                isinstance(regularization_factor, float) and regularization_factor >= 0
            )
            assert regularization_threshold is None or isinstance(
                regularization_threshold, float
            )

        self.eps = eps
        self.lr = lr
        self.model_loss = model_loss
        self.cond_fn = cond_fn
        self.optimizer_class = optimizer_class
        self.reg_factor = regularization_factor
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.reg_norm = (
            None
            if regularization_type is None
            else (1 if regularization_type == "l1" else 2)
        )
        self.regularization_threshold = regularization_threshold
        self.iter = 0
        self.eot_iters = eot_iters

    def reset(self, modifier):
        self.optimizer = self.optimizer_class(
            [{"params": m, "lr": self.lr[i]} for i, m in enumerate(modifier)],
        )
        self.iter = 0

    def __call__(self, x, modifier, ox, label, const, filt):
        self.iter = self.iter + 1
        outs = self.__run(x, modifier, ox, label, const, filt)
        loss = outs[-1]  # / self.eot_iters
        loss.backward()

        for i in range(len(self.optimizer.param_groups)):
            torch.nn.utils.clip_grad_norm_(
                self.optimizer.param_groups[i]["params"], 1.0
            )

        if self.iter % self.eot_iters == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.iter = 0
        return outs[:-1]

    def dist_fn(self, x, y):
        ret = self.cond_fn(x, y).view(-1)
        return ret, 1 - (ret.detach() < self.eps).float()

    def to_range(self, x):
        return x * (self.clip_max - self.clip_min) + self.clip_min

    def tanh(self, x):
        return self.to_range((torch.tanh(x) + 1) / 2)

    def norm_loss(self, x, y, norm):
        if self.reg_norm is None or self.reg_factor <= 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype), 0
        reg = (
            torch.pow(
                torch.norm(x - y, p=norm, dim=tuple(list(range(1, len((x).shape))))),
                1,
            )
        ).view(x.shape[0], -1)
        thresh = (
            self.regularization_threshold
            if self.regularization_threshold is not None
            else 0.0
        )
        return reg, 1 - (reg.detach() < thresh).float()

    def __run(self, x, modifier, ox, label, const, filt):
        new_x = filt(self.tanh(modifier + x), ox)
        dist_loss, dist_idx = self.dist_fn(new_x, ox)
        filter_loss = filt.compute_loss(new_x, guidance=ox, normalize=False)
        label_loss = self.model_loss(new_x, label).view(-1)
        reg, reg_idx = self.norm_loss(new_x, ox, self.reg_norm)
        total_reg = reg * reg_idx * self.reg_factor
        total_dist_loss = dist_loss * dist_idx * const
        loss = (total_dist_loss + label_loss + filter_loss + total_reg).sum()
        return (
            new_x,
            dist_loss.detach(),
            label_loss.detach(),
            reg.detach(),
            filter_loss.detach(),
            loss,
        )


class Bar:
    def __init__(
        self,
        model,
        itrs,
        binary_search_steps,
        targeted=False,
        with_filter_loss=True,
        with_reg=True,
    ):
        self.itrs = itrs
        self.pbar = trange(self.itrs, leave=False)
        self.binary_search_steps = binary_search_steps
        self.step = 0
        self.model = model
        self.itr = 0
        self.with_filter_loss = with_filter_loss
        self.with_reg = with_reg
        self.targeted = targeted
        self.global_reset()

    def reset(self):
        self.itr = 0
        self.step += 1
        self.pbar.reset()

    def global_reset(self):
        self.reset()
        self.step = 0
        self.pbar.set_description("LF progress")

    def close(self):
        self.pbar.update(self.itrs - self.itr)
        self.global_reset()

    def update(
        self,
        best_loss,
        x,
        label,
        dist_loss,
        label_loss,
        filter_loss,
        reg_loss,
        do_update=False,
    ):
        self.itr = (self.itr + 1) if do_update else self.itr
        best_loss = round(best_loss.detach().mean().item(), 4)
        label_loss = round(label_loss.detach().mean().item(), 4)
        dist_loss = round(dist_loss.mean().detach().item(), 4)
        Message = f"LF progress - Step {self.step}/{self.binary_search_steps}, best loss: {best_loss}, curr loss: {label_loss}, dist: {dist_loss}"
        if self.with_reg:
            Message = Message + f", reg_loss: {round(float(reg_loss.mean().item()), 4)}"
        if self.with_filter_loss:
            Message = (
                Message
                + f", filter_loss: {round(float(filter_loss.mean().detach().item()), 4)}"
            )
        self.pbar.set_description(Message)
        if do_update:
            with torch.no_grad():
                _, eval_loss, eval_loss_single = self.model.eval_attack(
                    x.detach(),
                    label,
                    targeted=self.targeted,
                )
            self.pbar.update(1)
            return eval_loss, eval_loss_single
        return 0.0, 0.0


class LF:
    def __init__(
        self,
        model,
        dist_fn,
        eps,
        optimizer_args,
        targeted=False,
        filter_args=None,
        max_iterations=1000,
        binary_search_steps=1,
        eot_iters=1,
        initial_const=1e-2,
        clip_min=0.0,
        clip_max=1.0,
        abort_early=True,
    ):
        self.model = model
        loss = model.get_loss_fn()
        assert isinstance(dist_fn, dict) and "class" in list(dist_fn.keys())
        initial_const = float(initial_const)
        assert isinstance(binary_search_steps, int) and binary_search_steps > 0
        assert isinstance(eot_iters, int) and eot_iters > 0
        assert isinstance(max_iterations, int) and max_iterations > 0
        assert filter_args is None or isinstance(filter_args, dict)
        assert (
            optimizer_args.get("learning_rate") is not None
            and isinstance(optimizer_args["learning_rate"], dict)
            and optimizer_args["learning_rate"].get("values") is not None
        )
        filter_args = {} if filter_args is None else filter_args
        kernels = filter_args.get("kernels", [])
        box = filter_args.get("box", (1, 1))
        learning_rates = optimizer_args["learning_rate"]["values"]
        isinstance(learning_rates, list)
        assert isinstance(kernels, list)
        if len(learning_rates) != len(kernels) + 1:
            assert len(learning_rates) == 2
            learning_rates = ([learning_rates[0]] * len(kernels)) + [learning_rates[1]]
        for lr in learning_rates:
            assert isinstance(lr, float)
        reg_args = (
            {}
            if optimizer_args.get("regularization", {}) is None
            else optimizer_args.get("regularization", {})
        )
        assert isinstance(reg_args, dict)
        self.__check_kernels(kernels, box)
        with_filter_loss = not (
            len(kernels) == 0 or filter_args.get("loss_factor", 1) == 0
        )
        with_reg = reg_args.get("type") is not None and reg_args.get("factor") > 0
        if "kernels" in list(filter_args.keys()):
            del filter_args["kernels"]
        optimizer_args = {
            "optimizer_class": SGNOpt if optimizer_args.get("grad_sgn", True) else Adam,
            "regularization_type": reg_args.get("type", None),
            "regularization_factor": reg_args.get("factor", 0.0),
            "regularization_threshold": reg_args.get("thresh"),
        }

        self.initial_const = initial_const
        self.kernels = kernels
        self.filter_args = filter_args
        self.abort_early = abort_early

        self.optimizer = StepPerformer(
            LossAndModelWrapper(model, loss),
            eval(dist_fn["class"])(**dist_fn.get("args", {})).eval(),
            eps,
            learning_rates,
            eot_iters=eot_iters,
            clip_min=clip_min,
            clip_max=clip_max,
            **optimizer_args,
        )

        self.pbar = Bar(
            model,
            max_iterations,
            binary_search_steps,
            targeted=targeted,
            with_filter_loss=with_filter_loss,
            with_reg=with_reg,
        )
        self.targeted = targeted

        self.cond_fn = lambda x, y: (x < y)
        self.inf = INF

    def __check_kernels(self, kernels, box):
        for k in kernels:
            assert (isinstance(k, tuple) or isinstance(k, list)) and len(k) == 2
            for kx in k:
                assert isinstance(kx, int) and kx % 2 == 1
        if len(kernels) > 0:
            assert box is None or isinstance(box, tuple) or isinstance(box, list)
            if isinstance(box, tuple) or isinstance(box, list):
                assert len(box) == 2
                for bx in box:
                    assert isinstance(bx, int)

    def setup(self, shape, device, dtype):
        self.pbar.global_reset()
        self.optimizer.cond_fn = self.optimizer.cond_fn.to(device)
        self.lower_bound = torch.zeros((shape[0]), device=device).float()
        self.upper_bound = torch.ones((shape[0]), device=device).float() * 1e10
        self.const = (
            torch.ones((shape[0]), device=device, dtype=dtype) * self.initial_const
        )
        filt = Filter(
            self.kernels,
            shape,
            **self.filter_args,
        ).to(device)
        modifier = torch.zeros(
            shape,
            requires_grad=True,
            device=device,
            dtype=dtype,
        )
        self.optimizer.reset(list(filt.parameters()) + [modifier])
        return modifier, filt

    def update_const(self, bestscore):
        u_cond = torch.all(bestscore.view(bestscore.shape[0], -1) != -1, -1).float()
        self.upper_bound = torch.minimum(
            self.upper_bound, self.const
        ) * u_cond + self.upper_bound * (1 - u_cond)
        self.lower_bound = (
            torch.maximum(self.lower_bound, self.const) * (1 - u_cond)
            + self.lower_bound * u_cond
        )
        const_cond = (self.upper_bound < 1e9).float()
        self.const = (
            (self.lower_bound + self.upper_bound) / 2
        ) * const_cond + self.const * (
            10 * (1 - u_cond) * (1 - const_cond) + (1 - const_cond) * u_cond
        )

    def eval(self):
        self.model = self.model.eval()
        self.optimizer.cond_fn = self.optimizer.cond_fn.eval()
        return self

    def to(self, device):
        self.model = self.model.to(device)
        self.optimizer.cond_fn = self.optimizer.cond_fn.to(device)
        return self

    def __call__(self, x, label, y_init):
        self.model.set_y_orig(y_init)
        found = 0.0
        single_found = 0.0
        x = torch.clamp(x, self.optimizer.clip_min, self.optimizer.clip_max)
        x = (x - self.optimizer.clip_min) / (
            self.optimizer.clip_max - self.optimizer.clip_min
        )
        ox = x.detach().clone()
        o_bestattack = x.detach().clone()
        best_found = x.detach().clone()
        o_best_loss = torch.ones((x.shape[0])).float().to(x.device) * self.inf
        x = torch.arctanh((torch.clamp(x, 0, 1) * 2 - 1) * 0.999999).detach()
        modifier, filt = self.setup(x.shape, x.device, x.dtype)

        for outer_step in range(self.pbar.binary_search_steps):
            self.pbar.reset()
            best_loss = torch.ones((len(x))).float().to(x.device) * self.inf
            best_dist = torch.ones(len(x)).to(torch.float).to(x.device) * (-1.0)

            for i in range(self.pbar.itrs * self.optimizer.eot_iters):
                new_x, dist_loss, label_loss, reg_loss, filter_loss = self.optimizer(
                    x, modifier, ox, label, self.const, filt
                )
                succeeded_dist = dist_loss < self.optimizer.eps
                cond = torch.logical_and(
                    self.cond_fn(label_loss, o_best_loss), succeeded_dist
                )
                n_cond, cond = (
                    torch.logical_or(
                        cond,
                        torch.logical_and(
                            self.cond_fn(label_loss, best_loss), succeeded_dist
                        ),
                    ).float(),
                    cond.float(),
                )

                if (i + 1) % self.optimizer.eot_iters == 0:
                    o_best_loss = label_loss * cond + torch.nan_to_num(
                        o_best_loss * (1 - cond), nan=0.0
                    )
                    o_bestattack = new_x * unsqueeze_like(
                        cond, new_x
                    ) + o_bestattack * (1 - unsqueeze_like(cond, new_x))
                    best_loss = label_loss * n_cond + torch.nan_to_num(
                        best_loss * (1 - n_cond), nan=0.0
                    )
                    best_dist = dist_loss * n_cond + best_dist * (1 - n_cond)

                eval_result, eval_result_single = self.pbar.update(
                    best_loss,
                    o_bestattack.detach(),
                    label,
                    dist_loss,
                    label_loss,
                    filter_loss,
                    reg_loss,
                    do_update=(i + 1) % self.optimizer.eot_iters == 0,
                )
                if eval_result_single > 0:
                    single_found = 1.0
                    best_found_single = o_bestattack.detach()
                if eval_result > 0:
                    found = 1.0
                    best_found = o_bestattack.detach()
                    if self.abort_early:
                        self.pbar.close()
                        return o_bestattack.detach(), eval_result, eval_result
            self.update_const(best_dist)
            if found != 1.0 and single_found == 1.0:
                best_found = best_found_single
        self.pbar.close()
        return best_found, found, single_found
