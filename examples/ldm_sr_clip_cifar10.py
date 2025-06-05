import torch
from torchvision import transforms
import open_clip

from DiffBreak import Registry, Runner
from DiffBreak.utils import SuperResolutionWrapper
from DiffBreak.utils.classifiers.torch.classifier import Classifier


class CLIPClassifier(Classifier):
    """Zero-shot CLIP classifier for CIFAR-10."""

    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        prompts = [f"a photo of a {c}" for c in classes]
        text_tokens = tokenizer(prompts)
        super().__init__(model)
        self.preprocess = preprocess
        self.text_tokens = text_tokens

    def to(self, device):
        self.text_tokens = self.text_tokens.to(device)
        return super().to(device)

    def forward(self, x):
        pil_images = [transforms.ToPILImage()(img.cpu()) for img in x]
        batch = torch.stack([self.preprocess(im) for im in pil_images]).to(x.device)
        with torch.no_grad():
            image_features = self.model.encode_image(batch)
            text_features = self.model.encode_text(self.text_tokens)
            logits = 100.0 * image_features @ text_features.T
        return logits


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "clip_sr_cifar10"
    dataset_name = "cifar10"
    attack_name = "LF"

    dataset = Registry.dataset(dataset_name)
    classifier = CLIPClassifier().to(device).eval()
    loss = Registry.default_loss(attack_name)

    defender = SuperResolutionWrapper(
        classifier,
        loss,
        cache_dir="cache",
        num_inference_steps=100,
        eta=1.0,
    )

    attack_params = Registry.attack_params(dataset_name, attack_name)

    exp_conf = Runner.setup(
        out_dir,
        attack_params=attack_params,
        targeted=False,
    )

    runner = Runner(exp_conf, dataset, defender, loss, dm_class=None).to(device).eval()
    runner.execute()


if __name__ == "__main__":
    main()
