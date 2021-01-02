import tensorflow as tf


class DenseNet:
    def __init__(self, residual_layer, growth_rate=32, num_classes=10):
        self.residual_layer = residual_layer
        self.residual_layer_list = list()
        self.growth_rate = growth_rate
        self.output_classes = num_classes

    def fc_layer(self, layer_input, scope='fc'):
        with tf.variable_scope(scope):
            layer = tf.keras.layers.Flatten()(layer_input)
            layer = tf.keras.layers.Dense(units=self.output_classes, use_bias=False)(layer)

        return layer

    def dense_bottleneck(self, layer_input, scope='bottleneck'):
        with tf.variable_scope(scope):
            x = tf.layers.batch_normalization(layer_input, training=True)
            x = tf.keras.activations.relu(x)
            x = tf.keras.layers.Conv2D(filters=self.growth_rate, kernel_size=1,
                                       strides=1, padding='same', use_bias=False)(x)
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.keras.activations.relu(x)
            layer = tf.keras.layers.Conv2D(filters=self.growth_rate, kernel_size=3,
                                           strides=1, padding='same', use_bias=False)(x)

        return layer

    def dense_block(self, block_input, dn_layers, scope='dense_blk'):
        with tf.variable_scope(scope):
            x = tf.layers.batch_normalization(block_input, training=True)
            block_input = tf.concat(values=[block_input, x], axis=3)
            for i in range(dn_layers-1):
                block = self.dense_bottleneck(block_input, scope='bottleneck_'+str(i))
                block_input = tf.concat([block_input, block], axis=3)

        return block

    def transition_block(self, block_input, scope='trans_blk'):
        with tf.variable_scope(scope):
            block = tf.layers.batch_normalization(block_input, training=True)
            block = tf.keras.layers.Conv2D(filters=self.growth_rate, kernel_size=1,
                                           strides=1, padding='same', use_bias=False)(block)
            block = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(block)

        return block

    def build(self, model_input):
        self.get_residual_layer()

        with tf.variable_scope('conv_1'):
            x = tf.keras.layers.Conv2D(filters=self.growth_rate, kernel_size=3,
                                       strides=1, padding='same', use_bias=False)(model_input)
            # x = tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='same')

            for lidx, lnum in enumerate(self.residual_layer_list):
                x = self.dense_block(x, dn_layers=lnum, scope='dense_blk_'+str(lidx))
                x = self.transition_block(x, scope='trans_blk'+str(lidx))

            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            model = self.fc_layer(x)

        return model

    def get_residual_layer(self):
        if self.residual_layer == 121:
            self.residual_layer_list = [6, 12, 24, 16]
        elif self.residual_layer == 169:
            self.residual_layer_list = [6, 12, 32, 32]
        elif self.residual_layer == 201:
            self.residual_layer_list = [6, 12, 48, 32]
        elif self.residual_layer == 264:
            self.residual_layer_list = [6, 12, 64, 48]
        else:
            raise ValueError('[DenseNet] number of residual layer is invalid, try 121, 169, 201, 264')
