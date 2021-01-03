import tensorflow as tf


class MobileNet:
    def __init__(self, num_classes):
        self.output_classes = num_classes
        # (64, 2, 1) denotes filter 64, strides 2, repeat 1
        self.block_arch = [(32, 1, 1),
                           (24, 2, 1),
                           (128, 1, 1),
                           (128, 2, 1),
                           (256, 1, 1),
                           (256, 2, 1),
                           (512, 1, 5),
                           (512, 2, 1),
                           (1024, 2, 1)]

    @staticmethod
    def depthwise_block(block_input,
                        filters,
                        strides,
                        scope):

        with tf.variable_scope(scope):
            x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides,
                                                padding='same', use_bias=False)(block_input)
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.keras.activations.relu(x)

            x = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1,
                                       padding='valid', use_bias=False)(x)
            x = tf.layers.batch_normalization(x, training=True)
            layer = tf.keras.activations.relu(x)

        return layer

    def fc_layer(self, layer_input, scope='fc'):
        with tf.variable_scope(scope):
            layer = tf.keras.layers.Flatten()(layer_input)
            layer = tf.keras.layers.Dense(units=self.output_classes)(layer)

        return layer

    def build(self, model_input):
        with tf.variable_scope('conv_0'):
            # Change strides to 2 for CIFAR10
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1,
                                       padding='same', use_bias=False)(model_input)
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.keras.activations.relu(x)

        for bid, [filters, strides, num_block] in enumerate(self.block_arch):
            for i in range(num_block):
                x = self.depthwise_block(x, filters, strides, scope='dwise_'+str(bid)+'_'+str(i))

        # x = tf.keras.layers.AveragePooling2D(pool_size=1)(x)

        model = self.fc_layer(x, scope='fc')

        return model
