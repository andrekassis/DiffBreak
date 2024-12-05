import os
import gc
import warnings
from numbers import Number
import numpy as np
import torch
import functools
from torch.autograd import Variable, Function
import torch.nn.functional as F

from .utils import LossAndModelWrapper


class PerturbationNormLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        assert kwargs.get("lp", 2) in [1, 2, "inf"]
        self.lp = kwargs.get("lp", 2)

    def forward(self, examples, *args, **kwargs):
        return kwargs["perturbation"].perturbation_norm(lp_style=self.lp)


class RegularizedLoss(torch.nn.Module):
    def __init__(self, losses, scalars, negate=False):
        super().__init__()
        assert sorted(losses.keys()) == sorted(scalars.keys())
        self.losses = losses
        self.scalars = scalars
        self.negate = negate

    def forward(self, examples, labels, *args, **kwargs):
        output_per_example = kwargs.get("output_per_example", False)
        output = 0.0
        for k in self.losses:
            loss = self.losses[k]
            scalar = self.scalars[k]
            loss_val = loss(examples, labels, *args, **kwargs)
            addendum = loss_val * scalar
            if addendum.numel() > 1:
                if not output_per_example:
                    addendum = torch.sum(addendum)
            output = output + addendum
        return output * (-1) if self.negate else output


# assert initialized decorator
def initialized(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        assert self.initialized, (
            "Parameters not initialized yet. " "Call .forward(...) first"
        )
        return func(self, *args, **kwargs)

    return wrapper


def clamp_ref(x, y, l_inf):
    """Clamps each element of x to be within l_inf of each element of y"""
    return torch.clamp(x - y, -l_inf, l_inf) + y


def safe_tensor(entity):
    """Returns a tensor of an entity, which may or may not already be a
    tensor
    """
    warnings.warn(
        "As of >=pytorch0.4.0 this is no longer necessary", DeprecationWarning
    )
    if isinstance(entity, Variable):
        return entity.data
    elif isinstance(entity, torch.tensor._TensorBase):
        return entity
    elif isinstance(entity, np.ndarray):
        return torch.Tensor(entity)  # UNSAFE CUDA CASTING
    else:
        raise Exception("Can't cast %s to a Variable" % entity.__class__.__name__)


def fold_mask(x, y, mask):
    """Creates a new tensor that's the result of masking between x and y
    ARGS:
        x : Tensor or Variable (NxSHAPE) - tensor that we're selecting where the
            masked values are 1
        y : Tensor or Variable (NxSHAPE) - tensor that we're selecting where the
            masked values are 0
        mask: ByteTensor (N) - masked values. Is only one dimensional: we expand
              it in the creation of this
    RETURNS:
        new object of the same shape/type as x and y
    """
    broadcast_mask = mask.view(-1, *tuple([1] * (x.dim() - 1)))
    broadcast_not_mask = (1 - safe_tensor(mask)).view(-1, *tuple([1] * (x.dim() - 1)))
    broadcast_not_mask = (
        Variable(broadcast_not_mask) if isinstance(x, Variable) else broadcast_not_mask
    )
    output = torch.zeros_like(x)
    output.add_(x * (broadcast_mask.type(x.type())))
    output.add_(y * (broadcast_not_mask.type(y.type())))
    return output


def random_from_lp_ball(tensorlike, lp, lp_bound, dim=0):
    """Returns a new object of the same type/shape as tensorlike that is
        randomly samples from the unit ball.

        NOTE THIS IS NOT A UNIFORM SAMPLING METHOD!
        (that's hard to implement, https://mathoverflow.net/a/9192/123034)

    ARGS:
        tensorlike : Tensor - reference object for which we generate
                     a new object of same shape/memory_location
        lp : int or 'inf' - which style of lp we use
        lp_bound : float - size of the L
        dim : int - which dimension is the 'keep dimension'
    RETURNS:
        new tensorlike where each slice across dim is uniform across the
        lp ball of size lp_bound
    """
    assert isinstance(lp, int) or lp == "inf"

    rand_direction = torch.rand(tensorlike.shape).type(tensorlike.type())

    if lp == "inf":
        return rand_direction * (2 * lp_bound) - lp_bound
    else:
        rand_direction = rand_direction - 0.5  # allow for sign swapping
        # first magnify such that each element is above the ball
        min_norm = torch.min(batchwise_norm(rand_direction.abs(), lp, dim=dim))
        rand_direction = rand_direction / (min_norm + 1e-6)
        rand_magnitudes = torch.rand(tensorlike.shape[dim]).type(tensorlike.type())
        rand_magnitudes = rand_magnitudes.unsqueeze(1)
        rand_magnitudes = rand_magnitudes.expand(*rand_direction.shape)

        return torch.renorm(rand_direction, lp, dim, lp_bound) * rand_magnitudes


def batchwise_norm(examples, lp, dim=0):
    """Returns the per-example norm of the examples, keeping along the
        specified dimension.
        e.g. if examples is NxCxHxW, applying this fxn with dim=0 will return a
             N-length tensor with the lp norm of each example
    ARGS:
        examples : tensor or Variable -  needs more than one dimension
        lp : string or int - either 'inf' or an int for which lp norm we use
        dim : int - which dimension to keep
    RETURNS:
        1D object of same type as examples, but with shape examples.shape[dim]
    """

    assert isinstance(lp, int) or lp == "inf"
    examples = torch.abs(examples)
    example_dim = examples.dim()
    if dim != 0:
        examples = examples.transpose(dim, 0)

    if lp == "inf":
        for reduction in range(1, example_dim):
            examples, _ = examples.max(1)
        return examples

    else:
        examples = torch.pow(examples + 1e-10, lp)
        for reduction in range(1, example_dim):
            examples = examples.sum(1)
        return torch.pow(examples, 1.0 / lp)


class FullSpatial(torch.nn.Module):
    def __init__(self, shape, use_gpu=True):
        """FullSpatial just has parameters that are the grid themselves.
        Forward then will just call grid sample using these params directly
        """

        super().__init__()
        self.use_gpu = use_gpu
        self.img_shape = shape
        self.xform_params = torch.nn.Parameter(self.identity_params(shape))

    def identity_params(self, shape):
        """Returns some grid parameters such that the minibatch of images isn't
            changed when forward is called on it
        ARGS:
            shape: torch.Size - shape of the minibatch of images we'll be
                   transforming. First index should be num examples
        RETURNS:
            torch TENSOR (not variable!!!)
            if shape arg has shape NxCxHxW, this has shape NxCxHxWx2
        """

        # Work smarter not harder -- use idenity affine transforms here
        num_examples = shape[0]
        identity_affine_transform = torch.zeros(num_examples, 2, 3)
        if self.use_gpu:
            identity_affine_transform = identity_affine_transform.cuda()

        identity_affine_transform[:, 0, 0] = 1
        identity_affine_transform[:, 1, 1] = 1

        return F.affine_grid(
            identity_affine_transform, shape, align_corners=True
        ).data  # F.affine_grid(identity_affine_transform, shape).data

    def stAdv_norm(self):
        """Computes the norm used in
        "Spatially Transformed Adversarial Examples"
        """

        # ONLY WORKS FOR SQUARE MATRICES
        dtype = self.xform_params.data.type()
        num_examples, height, width = tuple(self.xform_params.shape[0:3])
        assert height == width
        ######################################################################
        #   Build permutation matrices                                       #
        ######################################################################

        def id_builder():
            x = torch.zeros(height, width).type(dtype)
            for i in range(height):
                x[i, i] = 1
            return x

        col_permuts = []
        row_permuts = []
        # torch.matmul(foo, col_permut)
        for col in ["left", "right"]:
            col_val = {"left": -1, "right": 1}[col]
            idx = (torch.arange(width) - col_val) % width
            idx = idx.type(dtype).type(torch.LongTensor)
            if self.xform_params.is_cuda:
                idx = idx.cuda()

            col_permut = torch.zeros(height, width).index_copy_(
                1, idx.cpu(), id_builder().cpu()
            )
            col_permut = col_permut.type(dtype)

            if col == "left":
                col_permut[-1][0] = 0
                col_permut[0][0] = 1
            else:
                col_permut[0][-1] = 0
                col_permut[-1][-1] = 1
            col_permut = Variable(col_permut)
            col_permuts.append(col_permut)
            row_permuts.append(col_permut.transpose(0, 1))

        ######################################################################
        #   Build delta_u, delta_v grids                                     #
        ######################################################################
        id_params = Variable(self.identity_params(self.img_shape))
        delta_grids = self.xform_params - id_params
        delta_grids = delta_grids.permute(0, 3, 1, 2)

        ######################################################################
        #   Compute the norm                                                 #
        ######################################################################
        output = Variable(torch.zeros(num_examples).type(dtype))

        for row_or_col, permutes in zip(["row", "col"], [row_permuts, col_permuts]):
            for permute in permutes:
                if row_or_col == "row":
                    temp = delta_grids - torch.matmul(permute, delta_grids)
                else:
                    temp = delta_grids - torch.matmul(delta_grids, permute)
                temp = temp.pow(2)
                temp = temp.sum(1)
                temp = (temp + 1e-10).pow(0.5)
                output.add_(temp.sum((1, 2)))
        return output

    def norm(self, lp="inf"):
        """Returns the 'norm' of this transformation in terms of an LP norm on
            the parameters, summed across each transformation per minibatch
        ARGS:
            lp : int or 'inf' - which lp type norm we use
        """

        if isinstance(lp, int) or lp == "inf":
            identity_params = Variable(self.identity_params(self.img_shape))
            return batchwise_norm(self.xform_params - identity_params, lp, dim=0)
        else:
            assert lp == "stAdv"
            return self._stAdv_norm()

    def clip_params(self):
        """Clips the parameters to be between -1 and 1 as required for
        grid_sample
        """
        clamp_params = torch.clamp(self.xform_params, -1, 1).data
        change_in_params = clamp_params - self.xform_params.data
        self.xform_params.data.add_(change_in_params)

    def merge_xform(self, other, self_mask):
        """Takes in an other instance of this same class with the same
        shape of parameters (NxSHAPE) and a self_mask bytetensor of length
        N and outputs the merge between self's parameters for the indices
        of 1s in the self_mask and other's parameters for the indices of 0's
        """
        assert self.__class__ == other.__class__
        self_params = self.xform_params.data
        other_params = other.xform_params.data
        assert self_params.shape == other_params.shape
        assert self_params.shape[0] == self_mask.shape[0]
        assert other_params.shape[0] == self_mask.shape[0]

        new_xform = FullSpatial(shape=self.img_shape, use_gpu=self.use_gpu)
        new_params = fold_mask(
            self.xform_params.data, other.xform_params.data, self_mask
        )
        new_xform.xform_params = torch.nn.Parameter(new_params)
        return new_xform

    def project_params(self, lp, lp_bound):
        """Projects the params to be within lp_bound (according to an lp)
            of the identity map. First thing we do is clip the params to be
            valid, too
        ARGS:
            lp : int or 'inf' - which LP norm we use. Must be an int or the
                 string 'inf'
            lp_bound : float - how far we're allowed to go in LP land
        RETURNS:
            None, but modifies self.xform_params
        """

        assert isinstance(lp, int) or lp == "inf"

        # clip first
        self.clip_params()

        # then project back

        if lp == "inf":
            identity_params = self.identity_params(self.img_shape)
            clamp_params = clamp_ref(self.xform_params.data, identity_params, lp_bound)
            change_in_params = clamp_params - self.xform_params.data
            self.xform_params.data.add_(change_in_params)
        else:
            raise NotImplementedError("Only L-infinity bounds working for now ")

    def forward(self, x):
        # usual forward technique
        return F.grid_sample(
            x, self.xform_params, align_corners=True
        )  # F.grid_sample(x, self.xform_params)


class PGD(object):
    def __init__(
        self,
        classifier_net,
        normalizer,
        threat_model,
        loss_fxn,
        eval_ivl=-1,
        use_gpu=False,
    ):
        self.classifier_net = classifier_net
        self.normalizer = normalizer
        self.threat_model = threat_model
        self.loss_fxn = loss_fxn
        self.eval_ivl = eval_ivl
        self.use_gpu = use_gpu
        self.threat_model.use_gpu = use_gpu

    def to(self, device):
        if "cuda" in device:
            self.use_gpu = True
            self.threat_model.use_gpu = True
        elif "cpu" in device:
            self.use_gpu = False
            self.threat_model.use_gpu = False
        return self

    @property
    def _dtype(self):
        return torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor

    def setup(self):
        self.classifier_net.eval()
        self.normalizer.differentiable_call()

    def attack(
        self,
        examples,
        labels,
        targeted=True,
        step_size=1.0 / 255.0,
        num_iterations=20,
        random_init=False,
        signed=True,
        optimizer=None,
        optimizer_kwargs=None,
        loss_convergence=0.999,
        verbose=True,
        keep_best=True,
        eot_iter=1,
    ):
        """Builds PGD examples for the given examples with l_inf bound and
            given step size. Is almost identical to the BIM attack, except
            we take steps that are proportional to gradient value instead of
            just their sign.

        ARGS:
            examples: NxCxHxW tensor - for N examples, is NOT NORMALIZED
                      (i.e., all values are in between 0.0 and 1.0)
            labels: N longTensor - single dimension tensor with labels of
                    examples (in same order as examples)
            l_inf_bound : float - how much we're allowed to perturb each pixel
                          (relative to the 0.0, 1.0 range)
            step_size : float - how much of a step we take each iteration
            num_iterations: int or pair of ints - how many iterations we take.
                            If pair of ints, is of form (lo, hi), where we run
                            at least 'lo' iterations, at most 'hi' iterations
                            and we quit early if loss has stabilized.
            random_init : bool - if True, we randomly pick a point in the
                               l-inf epsilon ball around each example
            signed : bool - if True, each step is
                            adversarial = adversarial + sign(grad)
                            [this is the form that madry et al use]
                            if False, each step is
                            adversarial = adversarial + grad
            keep_best : bool - if True, we keep track of the best adversarial
                               perturbations per example (in terms of maximal
                               loss) in the minibatch. The output is the best of
                               each of these then
        RETURNS:
            AdversarialPerturbation object with correct parameters.
            Calling perturbation() gets Variable of output and
            calling perturbation().data gets tensor of output
        """

        ######################################################################
        #   Setups and assertions                                            #
        ######################################################################

        self.classifier_net.eval()
        perturbation = self.threat_model(examples)

        num_examples = examples.shape[0]
        var_examples = Variable(examples, requires_grad=True)
        var_labels = Variable(labels, requires_grad=False)

        if isinstance(num_iterations, int):
            min_iterations = num_iterations
            max_iterations = num_iterations
        elif isinstance(num_iterations, tuple):
            min_iterations, max_iterations = num_iterations

        verbose = True

        best_perturbation = None
        if keep_best:
            best_loss_per_example = {i: None for i in range(num_examples)}

        prev_loss = None

        single_found = False
        best_single_pert = None

        ######################################################################
        #   Loop through iterations                                          #
        ######################################################################

        # random initialization if necessary
        if random_init:
            perturbation.random_init()

        # Build optimizer techniques for both signed and unsigned methods
        optimizer = optimizer or optim.Adam
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 0.0001}
        optimizer = optimizer(perturbation.parameters(), **optimizer_kwargs)

        update_fxn = lambda grad_data: -1 * step_size * torch.sign(grad_data)

        param_list = list(perturbation.parameters())
        assert len(param_list) == 1, len(param_list)
        param = param_list[0]
        print(
            f"inside PGD attack, eot_iter: {eot_iter}, max_iterations: {max_iterations}"
        )
        for iter_no in range(max_iterations):
            perturbation.zero_grad()

            grad = torch.zeros_like(param)
            loss_per_example_ave = 0
            for i in range(eot_iter):
                loss_per_example = self.loss_fxn(
                    perturbation(var_examples),
                    var_labels,
                    perturbation=perturbation,
                    output_per_example=keep_best,
                )

                loss_per_example_ave += loss_per_example.detach().clone()
                loss = -1 * loss_per_example.sum()
                loss.backward()
                grad += param.grad.data.detach()
                param.grad.data.zero_()
                torch.cuda.empty_cache()
                gc.collect()

            grad /= float(eot_iter)
            loss_per_example_ave /= float(eot_iter)

            assert len(param_list) == 1, len(param_list)
            param.grad.data = grad.clone()

            if signed:
                perturbation.update_params(update_fxn)
            else:
                optimizer.step()

            if keep_best:
                mask_val = torch.zeros(num_examples, dtype=torch.uint8)
                for i, el in enumerate(loss_per_example_ave):
                    this_best_loss = best_loss_per_example[i]
                    if this_best_loss is None or this_best_loss[1] < float(el):
                        mask_val[i] = 1
                        best_loss_per_example[i] = (iter_no, float(el))

                if best_perturbation is None:
                    best_perturbation = self.threat_model(examples)

                best_perturbation = perturbation.merge_perturbation(
                    best_perturbation, mask_val
                )

            if self.eval_ivl > 0 and iter_no % self.eval_ivl == 0:
                with torch.no_grad():
                    print(f"############## iter: {iter_no} ##############")
                    _, successful_attack, single_succ = self.classifier_net.eval_attack(
                        perturbation(var_examples.detach()).detach(),
                        var_labels.detach(),
                        targeted=targeted,
                    )

                    if single_succ:
                        single_found = True
                        best_single_pert = perturbation

                    if float(successful_attack) > 0:
                        break

            # Stop early if loss didn't go down too much
            if (
                iter_no >= min_iterations
                and float(loss) >= loss_convergence * prev_loss
            ):
                if verbose:
                    print("Stopping early at %03d iterations" % iter_no)
                break
            prev_loss = float(loss)

        # perturbation = best_perturbation

        if single_found and not successful_attack:
            perturbation = best_single_pert
        perturbation.zero_grad()
        perturbation.attach_originals(examples)
        return perturbation, successful_attack, single_found


class ParameterizedXformAdv(torch.nn.Module):
    def __init__(self, threat_model, perturbation_params, use_gpu=True):
        super().__init__()
        self.threat_model = threat_model
        self.initialized = False
        self.perturbation_params = perturbation_params

        self.use_gpu = use_gpu
        self.lp_style = perturbation_params.lp_style
        self.lp_bound = perturbation_params.lp_bound
        self.use_stadv = perturbation_params.use_stadv
        self.scalar_step = perturbation_params.scalar_step or 1.0

    def _merge_setup(self, num_examples, new_xform):
        """DANGEROUS TO BE CALLED OUTSIDE OF THIS FILE!!!"""
        self.num_examples = num_examples
        self.xform = new_xform
        self.initialized = True

    def setup(self, originals):
        self.num_examples = originals.shape[0]
        self.xform = self.perturbation_params.xform_class(
            shape=originals.shape, use_gpu=self.use_gpu
        )
        self.initialized = True

    @initialized
    def perturbation_norm(self, x=None, lp_style=None):
        lp_style = lp_style or self.lp_style
        if self.use_stadv is not None:
            assert isinstance(self.xform, FullSpatial)
            return self.xform.stAdv_norm()
        else:
            return self.xform.norm(lp=lp_style)

    @initialized
    def constrain_params(self, x=None):
        # Do lp projections
        if isinstance(self.lp_style, int) or self.lp_style == "inf":
            self.xform.project_params(self.lp_style, self.lp_bound)

    @initialized
    def update_params(self, step_fxn):
        param_list = list(self.xform.parameters())
        assert len(param_list) == 1
        params = param_list[0]
        assert params.grad.data is not None
        self.add_to_params(step_fxn(params.grad.data) * self.scalar_step)

    @initialized
    def add_to_params(self, grad_data):
        """Assumes only one parameters object in the Spatial Transform"""
        param_list = list(self.xform.parameters())
        assert len(param_list) == 1
        params = param_list[0]
        params.data.add_(grad_data)

    @initialized
    def random_init(self):
        param_list = list(self.xform.parameters())
        assert len(param_list) == 1
        param = param_list[0]
        random_perturb = random_from_lp_ball(param.data, self.lp_style, self.lp_bound)

        param.data.add_(
            self.xform.identity_params(self.xform.img_shape)
            + random_perturb
            - self.xform.xform_params.data
        )

    @initialized
    def merge_perturbation(self, other, self_mask):
        assert self.__class__ == other.__class__
        assert self.threat_model == other.threat_model
        assert self.num_examples == other.num_examples
        assert self.perturbation_params == other.perturbation_params
        assert other.initialized
        new_perturbation = ParameterizedXformAdv(
            self.threat_model,
            self.perturbation_params,
            use_gpu=self.use_gpu,
        )

        new_xform = self.xform.merge_xform(other.xform, self_mask)
        new_perturbation._merge_setup(self.num_examples, new_xform)
        return new_perturbation

    @initialized
    def attach_originals(self, originals):
        """Little helper method to tack on the original images to self to
        pass around the (images, perturbation) in a single object
        """
        if hasattr(self, "originals"):
            raise Exception("%s already has attribute %s" % (self, "originals"))
        else:
            setattr(self, "originals", originals)

    def forward(self, x):
        if not self.initialized:
            self.setup(x)
        self.constrain_params()
        return self.xform.forward(x)

    def __call__(self, x):
        return self.forward(x)


class PerturbationParameters(dict):
    """Object that stores parameters like a dictionary.
        This allows perturbation classes to be only partially instantiated and
        then fed various 'originals' later.
    Implementation taken from : https://stackoverflow.com/a/14620633/3837607
    (and then modified with the getattribute trick to return none instead of
     error for missing attributes)
    """

    def __init__(self, *args, **kwargs):
        super(PerturbationParameters, self).__init__(*args, **kwargs)
        self.__dict__ = self
        self.use_gpu = kwargs.get("use_gpu", True)

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return None


class ThreatModel(object):
    def __init__(self, perturbation_class, param_kwargs):
        """Factory class to generate per_minibatch instances of Adversarial
            perturbations.
        ARGS:
            perturbation_class : class - subclass of Adversarial Perturbations
            param_kwargs : dict - dict containing named kwargs to instantiate
                           the class in perturbation class
        """
        self.perturbation_class = perturbation_class
        self.param_kwargs = PerturbationParameters(**param_kwargs)

    def __call__(self, *args):
        perturbation_obj = self.perturbation_class(
            self, self.param_kwargs, use_gpu=self.use_gpu
        )
        perturbation_obj.setup(*args)
        return perturbation_obj


class StAdvAttack(torch.nn.Module):
    def __init__(
        self,
        model,
        targeted=True,
        bound=0.05,
        num_iterations=100,
        lr=0.01,
        eot_iters=20,
        use_gpu=False,
        eval_interval=1,
    ):
        super().__init__()
        model_loss = model.get_loss_fn()
        threat_model = lambda: ThreatModel(
            ParameterizedXformAdv,
            {
                "lp_style": "inf",
                "lp_bound": bound,
                "xform_class": FullSpatial,
                "use_stadv": True,
            },
        )
        adv_loss = RegularizedLoss(
            {
                "cw": LossAndModelWrapper(model, model_loss),
                "pert": PerturbationNormLoss(lp=2),
            },
            {
                "cw": 1.0,
                "pert": 0.0025 / bound,
            },
            negate=True,
        )

        self.attack_kwargs = {
            "optimizer": torch.optim.Adam,
            "optimizer_kwargs": {"lr": lr},
            "signed": False,
            "verbose": False,
            "num_iterations": 100,
            "random_init": False,
            "eot_iter": eot_iters,
        }
        self.targeted = targeted
        self.attack = PGD(
            model,
            torch.nn.Identity(),
            threat_model(),
            adv_loss,
            eval_ivl=eval_interval,
            use_gpu=use_gpu,
        )
        self.model = model

    def eval(self):
        self.model = self.model.eval()
        return super().eval()

    def to(self, device):
        self.model = self.model.to(device)
        self.attack = self.attack.to(device)
        return super().to(device)

    def forward(self, inputs, labels, y_init):
        self.model.set_y_orig(y_init)
        pert, succ, single_succ = self.attack.attack(
            Variable(inputs).data, labels, targeted=self.targeted, **self.attack_kwargs
        )
        return pert(inputs).detach(), succ, float(single_succ)
