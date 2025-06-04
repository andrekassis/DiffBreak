import torch
from torchvision import transforms
from diffusers import LDMSuperResolutionPipeline

from .wrapper import ClassifierWrapper


class SuperResolutionWrapper(ClassifierWrapper):
    """Wrap a classifier with an LDM Super Resolution purification step."""

    def __init__(
        self,
        model_fn,
        model_loss,
        cache_dir="cache",
        num_inference_steps=100,
        eta=1.0,
        eval_mode="batch",
        verbose=0,
    ):
        super().__init__(model_fn, model_loss, eval_mode=eval_mode, verbose=verbose)
        self.cache_dir = cache_dir
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.pipeline = None

    def to(self, device):
        super().to(device)
        if self.pipeline is None:
            self.pipeline = LDMSuperResolutionPipeline.from_pretrained(
                "CompVis/ldm-super-resolution-4x-openimages",
                torch_dtype=torch.float16,
                cache_dir=self.cache_dir,
                local_files_only=True,
            ).to(device)
        else:
            self.pipeline = self.pipeline.to(device)
        return self

    def preprocess_eval(self, x):
        pil_images = [transforms.ToPILImage()(img.cpu()) for img in x]
        sr_images = []
        for im in pil_images:
            with torch.no_grad():
                sr = self.pipeline(
                    im, num_inference_steps=self.num_inference_steps, eta=self.eta
                ).images[0]
            sr_images.append(transforms.ToTensor()(sr))
        return torch.stack(sr_images, dim=0).to(x.device)

    def preprocess_forward(self, x, steps=None):
        return self.preprocess_eval(x)
