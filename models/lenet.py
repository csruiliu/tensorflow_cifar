import tensorflow as tf


class LeNet:
    def __init__(self, num_classes):
        self.output_classes = num_classes

    def build(self, model_input):
        x = tf.keras.layers.Conv2D(filters=6, kernel_size=5)(model_input)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
        x = tf.keras.layers.Conv2D(filters=16, kernel_size=5)(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=120)(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dense(units=84)(x)
        x = tf.keras.activations.relu(x)
        model = tf.keras.layers.Dense(units=self.output_classes)(x)

        return model
