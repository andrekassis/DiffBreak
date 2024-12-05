import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import keras
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model

from .classifier import Classifier, to_functional
from ...cache import load_classifier_from_cache


class RESNET50NP(Classifier):
    def __init__(self):
        path = path = load_classifier_from_cache("tf/youtube/resnet50np")
        model = keras.applications.ResNet50(
            classes=1283, weights=None, classifier_activation=None
        )
        model = Model(
            [model.input], Activation("softmax", dtype="float32")(model.output)
        )
        model = to_functional(model)
        model.load_weights(path)
        super().__init__(model, softmaxed=True)
