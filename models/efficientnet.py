import tensorflow as tf


class EfficientNet:
    def __init__(self, num_classes=10):
        # (expansion, in_filters, out_filters, kernel_size, repeats, strides)
        self.block_arch = [(1, 32, 16, 3, 1, 1),
                           (6, 16, 24, 3, 2, 2),
                           (6, 24, 40, 5, 3, 2),
                           (6, 40, 80, 3, 4, 2),
                           (6, 80, 112, 5, 3, 1),
                           (6, 112, 192, 5, 3, 2),
                           (6, 192, 320, 3, 1, 1)]

        self.output_classes = num_classes

    def fc_layer(self, layer_input, scope='fc'):
        with tf.variable_scope(scope):
            layer = tf.keras.layers.Flatten()(layer_input)
            layer = tf.keras.layers.Dense(units=self.output_classes)(layer)

        return layer

    @staticmethod
    def swish_activations(x_input):
        return x_input * tf.keras.activations.sigmoid(x_input)

    # squeeze-excitation layer
    def se_layer(self, layer_input, se_filters):
        x = tf.keras.layers.GlobalAveragePooling2D()(layer_input)
        # print(x.shape)
        x = tf.keras.layers.Conv2D(filters=se_filters, kernel_size=1)(layer_input)
        x = self.swish_activations(x)

        x = tf.keras.layers.Conv2D(filters=layer_input.shape[-1], kernel_size=1)(x)
        x = tf.keras.activations.sigmoid(x)
        layer = x * layer_input

        return layer

    # mobile inverted bottleneck
    def mbconv_block(self,
                     blk_input,
                     expansion,
                     in_filters,
                     out_filters,
                     kernel_size,
                     strides,
                     scope='mbc'):

        se_filters = expansion * in_filters
        with tf.variable_scope(scope):
            x = tf.keras.layers.DepthwiseConv2D(kernel_size, strides, padding='same', use_bias=False)(blk_input)
            x = tf.layers.batch_normalization(x, training=True)
            x = self.swish_activations(x)

            x = self.se_layer(x, se_filters)

            x = tf.keras.layers.Conv2D(out_filters, kernel_size=1, strides=1, use_bias=False)(x)
            layer = tf.layers.batch_normalization(x, training=True)

            if strides == 1 and in_filters == out_filters:
                dropout = tf.keras.layers.Dropout(rate=0.2)(layer)
                layer = layer + dropout

        return layer

    def build(self, model_input):
        with tf.variable_scope('conv1'):
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', use_bias=False)(model_input)
            x = tf.layers.batch_normalization(x, training=True)
            x = self.swish_activations(x)

        for bid, [expansion, in_filters, out_filters, kernel_size, num_block, strides] in enumerate(self.block_arch):
            for i in range(num_block):
                x = self.mbconv_block(x, expansion, in_filters, out_filters,
                                      kernel_size, strides, scope='mbc_'+str(bid)+'_'+str(i))

        x = tf.keras.layers.Conv2D(filters=1280, kernel_size=1, use_bias=False)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        model = self.fc_layer(x, scope='fc')

        return model
