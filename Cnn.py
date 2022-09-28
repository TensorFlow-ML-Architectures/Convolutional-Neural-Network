from keras import layers
from tensorflow import keras

class Cnn:
    def __int__(self):
        self.dims = None
        self.num_classes = None
        self.model = None
        self.inputs = None

    def model_create(self, dims, num_classes):
        self.dims = dims
        self.num_classes = num_classes
        self.inputs = keras.Input(shape=self.dims)

        # Image augmentation block
        rot = 0.1
        flip = "horizontal"
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip(flip),
                layers.RandomRotation(rot),
            ]
        )
        x = data_augmentation(self.inputs)

        # re-scale
        x = layers.Rescaling(1.0 / 255)(x)

        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        for size in [128, 256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        if self.num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = self.num_classes

        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(units, activation=activation)(x)

        self.model = keras.Model(self.inputs, outputs)

        self.model.summary()