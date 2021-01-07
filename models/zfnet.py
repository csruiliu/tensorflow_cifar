import tensorflow as tf


class ZFNet:
    def __init__(self, num_classes=10):
        self.output_classes = num_classes

    @staticmethod
    def fc_layer(layer_input, output_unit):
        layer = tf.keras.layers.Flatten()(layer_input)
        layer = tf.keras.layers.Dense(units=output_unit)(layer)

        return layer

    def build(self, model_input):
        with tf.variable_scope('conv_1'):
            x = tf.keras.layers.Conv2D(filters=96, kernel_size=7, strides=2, padding='same')(model_input)
            x = tf.keras.activations.relu(x)
            x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

        with tf.variable_scope('conv_2'):
            x = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding='same')(x)
            x = tf.keras.activations.relu(x)
            x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

        with tf.variable_scope('conv_3'):
            x = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same')(model_input)
            x = tf.keras.activations.relu(x)

        with tf.variable_scope('conv_4'):
            x = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same')(model_input)
            x = tf.keras.activations.relu(x)

        with tf.variable_scope('conv_5'):
            x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(model_input)
            x = tf.keras.activations.relu(x)
            x = tf.keras.layers.MaxPool2D(pool_size=3)(x)

        with tf.variable_scope('fc1'):
            x = self.fc_layer(x, output_unit=4096)
            x = tf.keras.activations.relu(x)
            x = tf.keras.layers.Dropout(rate=0.5)(x)

        with tf.variable_scope('fc2'):
            x = self.fc_layer(x, output_unit=4096)
            x = tf.keras.activations.relu(x)
            x = tf.keras.layers.Dropout(rate=0.5)(x)

        model = self.fc_layer(x, output_unit=self.output_classes)

        return model
