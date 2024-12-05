from keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
)
from keras.models import Sequential
from keras import regularizers

from .classifier import Classifier, to_functional
from ...cache import load_classifier_from_cache


class VGG16(Classifier):
    def __init__(self):
        path = load_classifier_from_cache("tf/cifar10/vgg16")
        activation = "softmax"
        momentum = 0.99
        weight_decay = 0.0005

        model = Sequential()
        model.add(
            Conv2D(
                64,
                (3, 3),
                padding="same",
                input_shape=(32, 32, 3),
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dropout(0.3))

        model.add(
            Conv2D(
                64,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=momentum))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Conv2D(
                128,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dropout(0.4))

        model.add(
            Conv2D(
                128,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=momentum))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Conv2D(
                256,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dropout(0.4))

        model.add(
            Conv2D(
                256,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dropout(0.4))

        model.add(
            Conv2D(
                256,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=momentum))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Conv2D(
                512,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dropout(0.4))

        model.add(
            Conv2D(
                512,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dropout(0.4))

        model.add(
            Conv2D(
                512,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=momentum))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Conv2D(
                512,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dropout(0.4))

        model.add(
            Conv2D(
                512,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dropout(0.4))

        model.add(
            Conv2D(
                512,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=momentum))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=momentum))

        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation(activation))
        model = to_functional(model)
        model.load_weights(path)
        super().__init__(model, softmaxed=True)
