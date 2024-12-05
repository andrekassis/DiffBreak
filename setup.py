import os
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

requirements = [
    "torch==2.2.2",
    "torchvision==0.17.2",
    "timm==0.9.16",
    "wandb==0.16.6",
    "lightning==2.1.0",
    "pytorch_optimizer==2.6.1",
    "pytorch_forecasting==1.0.0",
    "transformers==4.40.1",
    "diffusers==0.21.3",
    "accelerate==0.28.0",
    "einops==0.8.0",
    "pytorch-lightning==1.5.0",
    "lpips==0.1.4",
    "kornia==0.7.2",
    "open_clip_torch==2.7.0",
    "tensorflow==2.9.0",
    "ninja==1.11.1.1",
    "checksumdir==1.2.0",
    "cleverhans==4.0.0",
    "numpy==1.24.1",
    "rich==13.9.4",
    "robustbench @ git+https://github.com/RobustBench/robustbench.git@fix-download",
    "torchdiffeq==0.2.3",
    "torchsde==0.2.6",
    "huggingface_hub==0.25.0",
    "tqdm==4.67.1",
    "platformdirs",
]

dependency_links = [
    "https://download.pytorch.org/whl/cu121",
]


def post_install():
    conda_prefix = os.environ["CONDA_PREFIX"]
    env_vars = os.path.join(conda_prefix, "etc/conda/activate.d/env_vars.sh")
    os.system("yes | conda install -c conda-forge cudatoolkit=11.8.0")
    os.system(
        "yes | conda install cuda-nvcc=11.8 cuda-nvtx=11.8 cuda-libraries-dev=11.8 cuda-cupti=11.8 -c nvidia"
    )
    os.system("yes | conda install nvidia::cuda-cudart-dev=11.8")
    try:
        os.symlink(
            os.path.join(conda_prefix, "lib"),
            os.path.join(conda_prefix, "lib64"),
            target_is_directory=True,
        )
    except:
        pass
    os.makedirs(os.path.join(conda_prefix, "etc/conda/activate.d"), exist_ok=True)
    with open(env_vars, "w") as f:
        f.write(
            'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))\n'
        )
        f.write(
            "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH"
        )

    os.system("source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh")


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        post_install()


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)
        post_install()


class PostEggInfoCommand(egg_info):
    """Post-egg_info."""

    def run(self):
        egg_info.run(self)
        post_install()


setup(
    name="DiffBreak",
    version="0.0.1",
    python_requires=">=3.10",
    packages=["DiffBreak.resources.cache", "DiffBreak.resources.assets", "DiffBreak"],
    include_package_files=True,
    entry_points={"console_scripts": ["diffbreak=docs.registry:registry"]},
    install_requires=requirements,
    dependency_links=dependency_links,
    cmdclass={
        "install": PostInstallCommand,
        "develop": PostDevelopCommand,
        #'egg_info': PostEggInfoCommand,
    },
)
