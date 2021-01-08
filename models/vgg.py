import tensorflow as tf


class VGG:
    def __init__(self, conv_layer, num_classes=10):
        # [filter, num_layer]
        if conv_layer == 11:
            self.conv_layer_list = [[64, 1], [128, 1], [256, 2], [512, 2], [512, 2]]
        elif conv_layer == 13:
            self.conv_layer_list = [[64, 2], [128, 2], [256, 2], [512, 2], [512, 2]]
        elif conv_layer == 16:
            self.conv_layer_list = [[64, 2], [128, 2], [256, 3], [512, 3], [512, 3]]
        elif conv_layer == 19:
            self.conv_layer_list = [[64, 2], [128, 2], [256, 4], [512, 4], [512, 4]]
        else:
            raise ValueError('[VGG] number of residual layer is invalid, try 121, 169, 201, 264')

        self.output_classes = num_classes

    def fc_layer(self, layer_input, scope='fc'):
        with tf.variable_scope(scope):
            layer = tf.keras.layers.Flatten()(layer_input)
            layer = tf.keras.layers.Dense(units=self.output_classes)(layer)

        return layer

    def build(self, model_input):
        x = model_input
        for bid, [filters, num_layer] in enumerate(self.conv_layer_list):
            for lid in range(num_layer):
                with tf.variable_scope('convblk_' + str(bid) + '_' + str(lid)):
                    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')(x)
                    x = tf.layers.batch_normalization(x, training=True)
                    x = tf.keras.activations.relu(x)

            x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

        # x = tf.keras.layers.AveragePooling2D(pool_size=1, strides=1)(x)
        model = self.fc_layer(x, scope='fc')

        return model
