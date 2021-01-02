import tensorflow as tf


class MobileNetV2:
    def __init__(self, num_classes):
        self.output_classes = num_classes
        self.block_arch = [(1, 32, 16, 1, 1),
                           (6, 16, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
                           (6, 24, 32, 3, 2),
                           (6, 32, 64, 4, 2),
                           (6, 64, 96, 3, 1),
                           (6, 96, 160, 3, 2),
                           (6, 160, 320, 1, 1)]

    @staticmethod
    def bottleneck_block(block_input,
                         filters_in,
                         filters_out,
                         expansion,
                         strides,
                         scope):

        with tf.variable_scope(scope):
            x = tf.keras.layers.Conv2D(expansion*filters_in, kernel_size=1, strides=1,
                                       padding='valid', use_bias=False)(block_input)
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.nn.relu6(x)
            x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides,
                                                padding='same', use_bias=False)(x)
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.nn.relu6(x)
            x = tf.keras.layers.Conv2D(filters_out, kernel_size=1, strides=1,
                                       padding='valid', use_bias=False)(x)
            layer = tf.layers.batch_normalization(x, training=True)

            if strides == 1:
                shortcut = tf.keras.layers.Conv2D(filters_out, kernel_size=1, strides=1,
                                                  padding='valid', use_bias=False)(block_input)
                shortcut = tf.layers.batch_normalization(shortcut, training=True)
                layer = layer + shortcut

        return layer

    def build(self, model_input):
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1,
                                   padding='same', use_bias=False)(model_input)
        x = tf.layers.batch_normalization(x, training=True)

        for bid, [expansion, in_filter, out_filter, num_block, strides] in enumerate(self.block_arch):
            for i in range(num_block):
                x = self.bottleneck_block(x, in_filter, out_filter, expansion, strides,
                                          scope='bottleneck_'+str(bid)+'_'+str(i))
                in_filter = out_filter

        x = tf.keras.layers.Conv2D(filters=1280, kernel_size=1, strides=1,
                                   padding='valid', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=True)
        x = tf.nn.relu6(x)
        #x = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(x)
        x = tf.keras.layers.Flatten()(x)
        model = tf.keras.layers.Dense(units=self.output_classes, use_bias=False)(x)

        return model
