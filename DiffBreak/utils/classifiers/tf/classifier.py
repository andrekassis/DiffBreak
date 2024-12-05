import torch
import tensorflow as tf
from keras.layers import Input
from keras.models import Sequential, Model

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


class TFEngine(torch.autograd.Function):
    @staticmethod
    @tf.function
    def run_tf_fn(xf, tf_fn):
        return tf_fn(xf)

    @staticmethod
    def forward(ctx, x, tf_fn):
        xf = tf.Variable(x.detach().cpu().numpy())
        with tf.GradientTape() as tp:
            output = TFEngine.run_tf_fn(xf, tf_fn)
        ctx.xf, ctx.tp, ctx.device, ctx.output = xf, tp, x.device, output
        return torch.tensor(output.numpy()).to(x.device)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            torch.from_numpy(
                ctx.tp.gradient(
                    ctx.output,
                    ctx.xf,
                    output_gradients=[tf.constant(grad_output.detach().cpu().numpy())],
                ).numpy()
            ).to(ctx.device),
            None,
        )


def remove_tf_probs(model, softmaxed=False):
    if softmaxed:
        model.layers[-1].activation = tf.keras.layers.Lambda(lambda x: x)
        model.trainable = False
    return model


def to_functional(seqmodel):
    if not isinstance(seqmodel, Sequential):
        return seqmodel
    input_layer = Input(shape=seqmodel.layers[0].input_shape[1:])
    prev_layer = input_layer
    for layer in seqmodel.layers:
        layer._inbound_nodes = []
        prev_layer = layer(prev_layer)
    return Model([input_layer], [prev_layer])


class Classifier(torch.nn.Module):
    def __init__(self, model_fn, softmaxed=True):
        super().__init__()
        self.model_fn = remove_tf_probs(model_fn, softmaxed=softmaxed)
        self.model_fn.trainable = False
        self.diff_fn = TFEngine.apply

    def forward(self, x):
        return self.diff_fn(x.permute(0, 2, 3, 1), self.model_fn)
