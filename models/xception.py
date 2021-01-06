import tensorflow as tf


class Xception:
    def __init__(self, num_classes=10):
        self.output_classes = num_classes

    @staticmethod
    def res_block(block_input, filters, down_sample, num_layer, scope='blk'):
        with tf.variable_scope(scope):
            if down_sample:
                shortcut = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, use_bias=False)(block_input)
                shortcut = tf.layers.batch_normalization(shortcut)
            else:
                shortcut = block_input

            x = block_input
            for i in range(num_layer):
                x = tf.keras.activations.relu(x)
                x = tf.keras.layers.SeparableConv2D(filters, kernel_size=1, use_bias=False)(x)
                x = tf.layers.batch_normalization(x)

            # x = tf.keras.layers.MaxPool2D(pool_size=2, strides=1)(x)
            layer = x + shortcut

        return layer

    def fc_layer(self, layer_input, scope='fc'):
        with tf.variable_scope(scope):
            layer = tf.keras.layers.Flatten()(layer_input)
            layer = tf.keras.layers.Dense(units=self.output_classes)(layer)

        return layer

    def build(self, model_input):
        with tf.variable_scope('entry_flow'):
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=2, use_bias=False)(model_input)
            x = tf.layers.batch_normalization(x)
            x = tf.keras.activations.relu(x)

            with tf.variable_scope('blk_1'):
                shortcut = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, use_bias=False)(x)
                x = tf.keras.layers.SeparableConv2D(filters=128, kernel_size=1, padding='same', use_bias=False)(x)
                x = tf.layers.batch_normalization(x)
                x = tf.keras.activations.relu(x)
                x = tf.keras.layers.SeparableConv2D(filters=128, kernel_size=3, padding='same', use_bias=False)(x)
                x = tf.layers.batch_normalization(x)
                x = x + shortcut

            x = self.res_block(x, filters=256, down_sample=True, num_layer=2, scope='blk_2')
            x = self.res_block(x, filters=728, down_sample=True, num_layer=2, scope='blk_3')

        with tf.variable_scope('middle_flow'):
            for i in range(8):
                x = self.res_block(x, filters=728, down_sample=False, num_layer=3, scope='blk_'+str(i))

        with tf.variable_scope('exit_flow'):
            shortcut = tf.keras.layers.Conv2D(1024, kernel_size=1, strides=2, use_bias=False)(x)
            shortcut = tf.layers.batch_normalization(shortcut)

            x = tf.keras.activations.relu(x)
            x = tf.keras.layers.SeparableConv2D(728, kernel_size=3, padding='same', use_bias=False)(x)
            x = tf.layers.batch_normalization(x)

            x = tf.keras.activations.relu(x)
            x = tf.keras.layers.SeparableConv2D(1024, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
            x = tf.layers.batch_normalization(x)

            x = x + shortcut

            x = tf.keras.layers.SeparableConv2D(1536, kernel_size=3, use_bias=False)(x)
            x = tf.layers.batch_normalization(x)
            x = tf.keras.activations.relu(x)

            x = tf.keras.layers.SeparableConv2D(1536, kernel_size=3, use_bias=False)(x)
            x = tf.layers.batch_normalization(x)
            x = tf.keras.activations.relu(x)

            x = tf.keras.layers.GlobalAveragePooling2D()(x)

        model = self.fc_layer(x, scope='fc')

        return model
