## Welcome to DiffBreak
##### The first comprehensive toolkit for reliably evaluating diffusion-based adversarial purification (DBP)

Official PyTorch implementation of our paper:
[Unlocking The Potential of Adaptive Attacks on Diffusion-Based Purification](https://arxiv.org/abs/2411.16598).

Andre Kassis, Urs Hengartner, Yaoliang Yu

### Description
DiffBreak provides a reliable toolbox for assessing the robustness of DBP-based defenses against adversarial examples. It offers a modular extension that efficiently back-propagates the exact gradients through any DBP-based defense. All previous attempts to evaluate DBP suffered from implementation issues that led to a false sense of security. Hence, we aim for DiffBreak to become the new standard for such evaluations to ensure the credibility of future findings. DiffBreak also allows users to experiment with a variety of gradient approximation techniques previously explored in the literature that may be suitable for threat models wherein exact gradient calculation is infesible (e.g., due to time limitations). Furthermore, no existing adversarial robustness libraries offer attacks specifically optimized for performance against this memory and time-exhaustive defense. The implementations of current attacks (e.g., AutoAttack) do not allow for batch evaluations of multiple EOT samples at once, leading to severe performance degradation and significantly limiting the number of feasible EOT iterations. Worse yet, integrating DBP with the classifier and incorporating it into the attack code is not trivial and is naturally error-prone in the lack of a unified framework. Thus, current evaluations have been strictly limited to AutoAttack and PGD. That said, many other adversarial strategies exist and in our paper we specifically find that perceptual attacks (e.g., our low-frequency-- LF-- attack) pose far more severe threats to DBP. DiffBreak adapts the implementations of known attacks (see below for a comprehensive list) to DBP and allows users to efficiently evaluate the defense's robustness using increased EOT batch sizes. With DiffBreak, any PyTorch or TF classifier can be protected using existing DBP schemes with any pretrained diffusion model and then evaluated against the attacks we offer. DiffBreak also allows evaluating non-defended (i.e., standard) classifiers.

### Acknowledgment
This repo was built on top of  [DiffPure](https://github.com/NVlabs/DiffPure). The adversarial attacks were adapted from various common libraries we cite below. If you consider our repo helpful, please consider citing it:

@article{kassis2024unlocking,
  title={Unlocking The Potential of Adaptive Attacks on Diffusion-Based Purification},
  author={Kassis, Andre and Hengartner, Urs and Yu, Yaoliang},
  journal={arXiv preprint arXiv:2411.16598},
  year={2024}
}

### Requirements
- A high-end NVIDIA GPU with >=32 GB memory.
- [CUDA=12](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html) driver must be installed.
- [Anaconda](https://docs.anaconda.com/anaconda/install/) must be installed.

### Installation
```console
conda create -n DiffBreak python=3.10
conda activate DiffBreak
git clone https://github.com/andrekassis/DiffBreak.git
cd DiffBreak
pip install -e .
```

After executing the above, a new command line tool ```diffbreak``` becomes available as well. You may use it to obtain information regarding the available configurations, datasets and pretrained systems we offer. To get started, just type ```diffbreak``` in your terminal. 

### Usage
Once DiffBreak has been installed, evaluating any classifier requires only a few lines of code. As mentioned above, we offer a variety of common pretrained classifiers (mostly taken from [Robustbench](https://github.com/RobustBench/robustbench) and [torchvision](https://pytorch.org/vision/stable/models.html))  for several datasets: [ImageNet](https://ieeexplore.ieee.org/document/5206848), [CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), [CelebA-HQ](https://arxiv.org/abs/1710.10196) and [YouTube-Faces](https://ieeexplore.ieee.org/document/5995566). You do not need to manually download any datasets or pretrained classifiers. Our code will automatically retrieve and cache these resources upon their first use. For datasets, their test subsets are used. To run evaluations with these readily-available resources, one should use DiffBreak's "Registry" utility. Alternatively, you may use your own datasets or models (see below). Evaluations require four components that are utilized by DiffBreak's "Runner" engine, which executes the experiments. These components are: 1) The dataset objects from which the samples are taken, 2) the classifier to be evaluated, 3) The DBP defense (only required if evaluating a DBP-defended classifier) and 4) the attack under which the system is evaluated. 

#### A) Initialization
Below we demonstrate how each of the four required components can be easily created.

##### A1) Dataset

To obtain an object that yields samples from a dataset of your choice offered by our registry, you should run:
```python
from DiffBreak import Registry
kwargs = {} #if using celeba-hq, kwargs = {"attribute": YOUR_TARGET_ATTRIBUTE}
dataset = Registry.dataset(dataset_name, **kwargs)
```
Here, <mark>**dataset_name**</mark> can be any of (case-sensitive):
- <mark>cifar10</mark>
- <mark>imagenet</mark>
- <mark>youtube</mark>
- <mark>celeba-hq</mark>: For this dataset, you must provide an additional keyword argument <mark>**attribute**</mark> as above (Run ```diffbreak datasets``` in you terminal for details). 

**You are not limited to the datasets in our registry. Refer to the output of ```diffbreak custom dataset```.**

##### A2) DBP
*<span style="color:gray">Skip this step if evaluating a non-defended classifier.</span>*

###### A2.1) Diffusion configuration
You should select the DBP scheme you intend to use for purification. We offer four known schemes for which DiffBreak provides the exact gradients:
- <mark>vpsde</mark>: The VP-SDE-Based DBP scheme (i.e., [DiffPure](https://arxiv.org/pdf/2205.07460)).
- <mark>vpode</mark>: Similar to <mark>vpsde</mark> but performs purification by solving an ODE in the reverse pass instead of an SDE. This method was also proposed in [DiffPure](https://arxiv.org/pdf/2205.07460).
- <mark>ddpm</mark>: The DDPM-based DBP scheme ([GDMP](https://arxiv.org/pdf/2205.14969)).
- <mark>ddpm_ode</mark>: Similar to <mark>vpode</mark> but implemented for discrete-time DBP (i.e., DDPM).

To initialize the gradient-enabled DBP defense, you should also specify the desired gradinet back-propagation method. The following methods are avialable:
- <mark>full</mark>: The full accurate gradients computed efficiently using our precise module (default).
- <mark>full_intermediate</mark>: Similar to <mark>full</mark> but additionally computes the loss function for each and every step of the reverse pass and adds its gradients to the backpropagated total.
- <mark>adjoint</mark>: The known [adjoint](https://arxiv.org/pdf/2001.01328) method for VP-based BDP (cannot be used with DDPM schemes). DiffBreak fixes the implementation issues of [torchsde](https://github.com/google-research/torchsde) and provides a far more powerful tool.
- <mark>bpda</mark>: [Backward-pass Differentiable Approximation (BPDA)](https://arxiv.org/pdf/1802.00420).
- <mark>blind</mark>: Gradients are obtained by attacking the classifier directly and without involving DBP at all. The defense is only considered when the attack sample is evaluated.
- <mark>forward_diff_only</mark>: Similar to <mark>blind</mark> but instead adds the noise from the forward pass of DBP to the sample and uses this noisy output to invoke the classifier and obtain the gradients.

With these choices, you can now invoke the registry to obtain a dictionary containing all the DBP parameters that DiffBreak expects for later initialization of the defense. To obtain the parameters, we run:
```python
dbp_params = Registry.dbp_params(
    dataset_name,
    diffusion_type=DBP_SCHEME,
    grad_mode=GRAD_MODE,
)
```
where DBP_SCHEME and GRAD_MODE are as explained above. Note that the returned dictionary contains standard DBP parameters used in the literature. Generally, you should not be change them (unless you know what you are doing). Exceptions are:
- <mark>**dbp_params**["diffusion_steps"]</mark>: The number of purification steps used in the defense. The returned values correspond to the optimal setups from the state-of-the-art but you may of course change them in your experiments.
- <mark>**dbp_params**["batch_size"]</mark>: The number of purified copies generated from the sample (EOT) that will be purified and classified in parallel. Change this based on you dataset dimentions and GPU capabilities if you wish or keep the default.
- <mark>**dbp_params**["timestep_respacing"]</mark>: Change this if you wish to perform DDPM acceleration.
- <mark>**dbp\_params**["guidance\_\*"]</mark>: Whether to perform guided purification (i.e., [GDMP](https://arxiv.org/pdf/2205.14969))-- Available only for DDPM variants. By default, DiffBreak performs guided purification for DDPM as in [GDMP](https://arxiv.org/pdf/2205.14969). To disable it, set <mark>**dbp\_params**["guidance\_mode"]=None</mark>. You may also change the remaining guidance parameters. See [GDMP](https://arxiv.org/pdf/2205.14969) for details.

###### A2.2) Score model
Now that you have the parameters, you should provide the score model to be used by DBP for purification. Our registry offers the following pretrained models:
- <mark>ScoreSDEModel</mark>: The Score SDE model by [Song et al.](https://arxiv.org/pdf/2011.13456). Default for <mark>cifar10</mark> with VP-based schemes.
- <mark>HaoDDPM</mark>: The model for CIFAR-10 by [Ho et al.](https://arxiv.org/pdf/2006.11239). Default for <mark>cifar10</mark> with DDPM schemes.
- <mark>DDPMModel</mark>: The CelebA-HQ pretrained model from [SDEdit](https://github.com/ermongroup/SDEdit). Default for <mark>celeba-hq</mark> and <mark>youtube</mark>.
- <mark>GuidedModel</mark>: The common guided model for ImageNet by [Dhariwal & Nichol](https://arxiv.org/pdf/2105.05233). Default for <mark>imagenet</mark>.

To instainitate one of these readily-available score models from the registry, we run the following command:
```python
dm_class = Registry.dm_class(dataset_name, diffusion_type=DBP_SCHEME)
```
with dataset_name and DBP_SCHEME as explained above. Note that you may use a different dataset_name or DBP_SCHEME here to obtain a different score model to use with your actual dataset and chosen scheme, provided that the score model operates on images of the same dimensions. That is, <mark>ScoreSDEModel</mark> and <mark>HaoDDPM</mark> may be used interchangeably for <mark>cifar10</mark>, while and <mark>DDPMModel</mark> and <mark>GuidedModel</mark> can be switched for all remaining datasets. 

**Importantly, you may also provide any external pretrained score model instead (run ```diffbreak custom dm_class``` for details).**

##### A3) Classifier
Our registry offers a variety of pretrained classifiers, which you can browse by executing ```diffbreak classifiers``` in your terminal. You can obtain the chosen classifier via:
```python
kwargs = {} #if using celeba-hq, kwargs = {"attribute": YOUR_TARGET_ATTRIBUTE}
classifier = Registry.classifier(
    dataset_name, classifier_name=CLASSIFIER_NAME, **kwargs
)
```
CLASSIFIER_NAME is the chosen architure you wish to use. You can also omit this argument to use the default classifier for your corresponding dataset: The ViT (DeiT-S) classifier by Facebook Research for <mark>imagenet</mark>, the WIDERESNET_70_16 classifier by [DiffPure](https://arxiv.org/pdf/2205.07460) for <mark>cifar10</mark>, the attribute classifiers by [gan-ensembling](https://github.com/chail/gan-ensembling) for <mark>celeba-hq</mark>, and a ResNet50 model we trained with TensorFlow for <mark>youtube</mark>.

Using pretrained classifiers not available in the registry is also possible. Instead of running the above code, wrap your own PyTorch classifier inside a DiffBreak classifier object:
```python
from DiffBreak import PyTorchClassifier
classifier = PyTorchClassifier(my_torch_classifier, softamaxed)
```
where <mark>**my_torch_classifier**</mark> is any such pretrained PyTorch classifier. It is also possible to use a TensorFlow (TF) classifier by executing:
```python
from DiffBreak import TFClassifier
classifier = TFClassifier(my_tf_classifier, softamaxed)
```
Here, <mark>**softmaxed**</mark> is a boolean indicating whether the last layer of the provided classifier applies a softmax activation or directly outputs the logits.

##### A4) Attack
DiffBreak offers a variety of attacks optimized for performance with DBP:
- <mark>id</mark>: No attack. Use this to evaluate clean accuracy.
- <mark>apgd</mark>: [AutoAttack (Linf)](https://arxiv.org/pdf/2003.01690).
-- Adapted from [auto-attack](https://github.com/fra31/auto-attack).
- <mark>pgd</mark>: The [PGD](https://arxiv.org/pdf/1607.02533) attack.
-- Adapted from [cleverhans](https://github.com/cleverhans-lab/cleverhans).
- <mark>diffattack_apgd</mark>: [DiffAttack](https://arxiv.org/pdf/2311.16124).
-- Adapted from [DiffAttack](https://github.com/kangmintong/DiffAttack).
- <mark>LF</mark>: Our [Low-Frequency](https://arxiv.org/pdf/2411.16598) attack.
-  <mark>diffattack_LF</mark>: Our  <mark>LF</mark> attack augmented with the per-step losses used by DiffAttack.
-  <mark>ppgd</mark>: [PerceptualPGDAttack](https://arxiv.org/abs/2006.12655).
-- Adapted from [perceptual-advex](https://github.com/cassidylaidlaw/perceptual-advex).
- <mark>lagrange</mark>: [LagrangePerceptualAttack](https://arxiv.org/abs/2006.12655).
-- Adapted from [perceptual-advex](https://github.com/cassidylaidlaw/perceptual-advex).
- <mark>stadv</mark>: The [StAdv](https://arxiv.org/pdf/1801.02612) attack.
-- Adapted from [perceptual-advex](https://github.com/cassidylaidlaw/perceptual-advex).

###### A4.1) Obtaining Attack Parameters
For each attack, the registry returns the default parameters for the corresponding dataset. We recommed <mark>LF</mark> and  <mark>ppgd</mark> against DBP-defended classifiers as we found them far more effective than the commonly-used norm-based methods (e.g., <mark>pgd</mark> and <mark>apgd</mark>). Obtaining the attack parameters from the registry is done as follows:
```python
attack_params = Registry.attack_params(dataset_name, attack_name)
```
where <mark>**attack_name**</mark> is one of the above options. For <mark>imagenet</mark>, <mark>cifar10</mark> and <mark>celeba-hq</mark>, these are the most commonly used parameters from the literature and you do not need to change them unless you explicitly intend to do so. 

The notable exception is <mark>**attack_params**["eot_iters"]</mark> that is universarlly present in the paramteres for all attacks. Changing this number will alter the effective number of EOT samples in your attack. Specifically, the total number of EOT samples will be <mark>**attack_params**["eot_iters"]</mark> \* <mark>**dbp_params**["batch_size"]</mark>. That is, <mark>**dbp_params**["batch_size"]</mark> samples are propagated at each one of the <mark>**attack_params**["eot_iters"]</mark> and their gradients are obtained. These are then added to the collective sum from all samples accross all<mark>**attack_params**["eot_iters"]</mark>, which is finally divided by the total number of samples above to get the averaged gradient. You may change this number based on your GPU capabilities or stick with the default.

###### A4.2) Selecting The Loss Function
The remaining component for initializing the attack is the choice of the loss function to optimize. Most attacks use similar loss functions and we provide these common choices in our registry. Specifically, the available losses are:
- <mark>CE</mark>: The cross-entropy loss. Default for <mark>pgd</mark>.
- <mark>MarginLoss</mark>: [The Max-Margin loss]( https://arxiv.org/pdf/1608.04644). Default for all remaining attacks.
- <mark>DLR</mark>: [The Difference of Logits Ratio loss](https://arxiv.org/pdf/2003.01690).
To obtain the default loss function for your chosen attack from the registry, run:
```python
loss = Registry.default_loss(attack_name)
```
where <mark>**attack_name**</mark> is the attack you intend to use. Note that AutoAttack (i.e., <mark>apgd</mark>) originally uses the <mark>DLR</mark> loss function. We instead use the <mark>MarginLoss</mark> as the default for this attack as we emprically found it yielding better results. 

You are not restricted to the default loss functions and may instantiate any of the available attacks with any of our provided losses. To directly obtain the loss function of your choice from DiffBreak instead, replace the above code, importing and instantiating the loss explicitly. For instance, to use <mark>DLR</mark> run:
```python
from DiffBreak import DLR
loss = DLR()
```

#### B) Running Evaluations
With the necessary objects now available, one can instantiate a "Runner" object and execute experiments. This is done over the two steps described below.

##### B1) Configuring The Runner
Each runner requires a configuration dictionary containing the attack and DBP parameters obtained above in a specific format, in addition to several other arguments speicifc to the expeirment itself. To construct this dictionary, the "Runner" class exposes a *setup* method with the below signature:
```python
setup(
    out_dir, attack_params, dbp_params=None, targeted=False, 
    eval_mode="batch", total_samples=256, 
	balanced_splits=False, verbose=2, seed=1234, 
	save_image_mode="originally_failed", overwrite=False
) -> dict
```
The parameters for this function are as follows:
- <mark>**out_dir**</mark>: **str**. The path to the output directory where the experiment results will be saved.
- <mark>**attack_params**</mark>: **dict**. The <mark>**attack_params**</mark> dictionary from A4.
- <mark>**dbp_params**</mark>: **dict** or None. If you wish to evaluate a DBP-defended classifier, this should be the <mark>**dbp_params**</mark> dictionary from A2. Otherwise, it should be None. <mark>Default: None</mark>.
-  <mark>**targeted**</mark>: **bool**. Whether you wish to perform a targeted attack. If True, a target label is drawn at random for each sample and the attack optimizes the input so that it is misclassified as belonging to this randomly chosen label. Otherwise, the standard non-targeted attack is performed with the objective of having the sample misclassified arbitrarily. <mark>Default: False</mark>.
-  <mark>**eval_mode**</mark>: **str** - one of <mark>batch</mark> or <mark>single</mark>. If <mark>single</mark>, the attack is considered successful for each sample if any purified copy is misclassified as desired. Otherwise, the attack is only considered successful if the majority of samples in the batch (depending on <mark>**dbp_params**["batch_size"]</mark>) meet this condition. This corresponds to the more robust "majority-vote" setup we study in our paper, while <mark>single</mark> represents the standard setup. For non-defended classifiers (i.e., <mark>**dbp_params**=None</mark>), this argument is ignored. <mark>Default: batch</mark>.
- <mark>**total_samples**</mark>: **int**. The total number of samples in the experiment. <mark>Default: 256</mark>.
- <mark>**balanced_splits**</mark>: **bool**. Whether to include an equal number of samples for all classes. <mark>Default: False</mark>.
- <mark>**verbose**</mark>: **int** - one of <mark>0</mark>, <mark>1</mark> or <mark>2</mark>. Verbosity level for logging. <mark>Default: 2</mark>. 
- <mark>**seed**</mark>: **int**. Random seed selected for reproducibility. <mark>Default: 1234</mark>.
- <mark>**save_image_mode**</mark>: **str** - one of <mark>none</mark>, <mark>successfull</mark> or <mark>originally_falied</mark>. Whether to save attack samples: <mark>none</mark> - no images will be saved. <mark>successfull</mark>: Only successful attack samples will be saved. <mark>originally_falied</mark>: Attack samples that are originally correctly classified but then misclassified with the attack will be saved, while samples that are initially misclassified (i.e., successfull attacks without adversarial modifications) will be skipped. <mark>Default: originally_falied</mark>.
- <mark>**overwrite**</mark>: **bool**. Whether to overwrite existing output directories. <mark>Default: False</mark>.

Constructing the configuration dictionary for your experiment is done by running:
```python
from DiffBreak import Runner
exp_conf = Runner.setup(
    out_dir,
    attack_params=attack_params,
    dbp_params=dbp_params,
    targeted=targeted,
    eval_mode=eval_mode,
    total_samples=total_samples,
    balanced_splits=balanced_splits,
    verbose=verbose,
    seed=seed,
    save_image_mode=save_image_mode,
	overwrite=overwrite,
)
```
with all parameters matching those described above.

**It is also possible to restore the experiment configuration from a previously started evaluation to resume it. This can be done with the *resume* method of the "Runner" class:**
```python
exp_conf = Runner.resume(out_dir)
```
where <mark>**out_dir**</mark> is the output directory of the previously started experiment. 

##### B2) Executing Experiments
At this stage, we have all the required components to run an experiment. First, we create a "Runner" instance as follows:
```python
device="cuda"
runner = Runner(exp_conf, dataset, classifier, loss, dm_class).to(device).eval()
```
where <mark>**exp_conf**</mark> is the configuration dictionary for the experiment obtained in the pervious step and <mark>**dataset**</mark>, <mark>**classifier**</mark>, <mark>**loss**</mark> and <mark>**dm_class**</mark> are the objects created in A1-A4. If you are evaluating a non-defended classifier, the argument <mark>**dm_class**</mark> can be omitted or alternatively must be set to <mark>**dm_class**=None</mark>. **Importantly, the runner must be moved to the chosen *cuda* device vefore it is used as shown above and the eval() method must be invoked.** 

Finally, run the experiment as:
```python
runner.execute()
```
The attack will be evaluated and the results will be saved to out_dir/results.txt. The output images will be saved to out_dir/images. During the evaluation, the progress bar constantly displays the portion of successfull attack samples. 

##### C) Understanding The Results

###### C1) Screen Logs
The runner constantly logs the "total success rate" to the screen which corresponds to the portion of samples that are misclassified in the desired manner (depending on whether the attack is targeted). For the <mark>**eval_mode**="batch"</mark>, "single success rate" is also printed-- this represents the protion of samples for which at least a single purified copy is misclassified (see B1). That is, <mark>**eval_mode**="batch"</mark> effectively evauates both setups (however it is far more costly than running with <mark>**eval_mode**="single"</mark>).

###### C2) results.txt File
The results.txt file is updated after the attack terminates for each sample, adding a row with the statistics corresponding to this input. Specifically, the n<sup>th</sup> row in this file contains the stats for the n<sup>th</sup> evaluated sample. These records are of the following format:
```console
original label: ORIG_LABEL, [target label: TARGET], originally robust: ORIG_ROBUST, result: SUCCESS, [result-single: SUCCESS_SINGLE].
```
The different entries in this record can be interpreted as follows:
- ORIG_LABEL: The sample's original label.
- [TARGET]: This entry only appears for targeted attacks. It represents the randomly chosen target label for the sample.
- ORIG_ROBUST: Assigned 0 or 1 depending on whether the classifier initially correctly classifies the sample (1) or not (0).
- SUCCESS: Assigned 0 or 1, indicating whether the attack was successful (1) or not (0).
- [SUCCESS_SINGLE]: This entry is only present for <mark>**eval_mode**="batch"</mark>. It indicates whether the attack is successful for this sample under the <mark>single</mark> evaluation mode as well.

##### D) Complete Example
Below, we combine all above steps to demonstrate how easily an evaluation can be created and executed. For this purpose, we show how to perform an experiment with the default classifier for <mark>cifar10</mark> (setting <mark>**classifier_name**=None</mark>) using our <mark>LF</mark> attack and the <mark>vpsde</mark> DBP scheme. All parameters are left identical to the defaults obtained from the registry. Running the experiment amounts to executing the short script below (other default arguments were excluded):

```python
from DiffBreak import Registry, Runner

device = "cuda"
out_dir = "test"
dataset_name = "cifar10"
attack_name = "LF"
dbp_scheme="vpsde"

dataset = Registry.dataset(dataset_name)
dm_class = Registry.dm_class(dataset_name, diffusion_type=dbp_scheme)
classifier = Registry.classifier(
    dataset_name, classifier_name=None,
)
loss = Registry.default_loss(attack_name)

attack_params = Registry.attack_params(dataset_name, attack_name)
dbp_params = Registry.dbp_params(
    dataset_name,
    diffusion_type=dbp_scheme,
    grad_mode="full",
)

exp_conf = Runner.setup(
    out_dir,
    attack_params=attack_params,
    dbp_params=dbp_params,
    targeted=False,
)

runner = (
    Runner(
        exp_conf,
        dataset,
        classifier,
        loss,
        dm_class,
    )
    .to(device)
    .eval()
)
runner.execute()
```


