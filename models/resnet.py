import tensorflow as tf


class ResNet:
    def __init__(self, residual_layer, num_classes):
        self.residual_layer = residual_layer
        self.residual_layer_list = list()
        self.output_classes = num_classes

    def fc_layer(self, layer_input, scope='fc'):
        with tf.variable_scope(scope):
            layer = tf.keras.layers.Flatten()(layer_input)
            layer = tf.keras.layers.Dense(units=self.output_classes)(layer)

        return layer

    # residual block is for resnet-18 and resnet-34
    @staticmethod
    def residual_block(block_input, filters, down_sample=False, block_name='conv'):
        with tf.variable_scope(block_name):
            if down_sample:
                x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding='same', use_bias=False)(block_input)
                shortcut = tf.keras.layers.Conv2D(filters, 1, strides=2, use_bias=False)(block_input)
                shortcut = tf.layers.batch_normalization(shortcut, training=True)
            else:
                x = tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(block_input)
                shortcut = block_input
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.keras.activations.relu(x)

            x = tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)
            x = tf.layers.batch_normalization(x, training=True)
            layer = tf.keras.activations.relu(x + shortcut)

        return layer

    # residual bottleneck is for resnet-50, resnet-101, and resnet-152
    @staticmethod
    def residual_bottleneck(block_input, filters, down_sample=False, block_name='conv'):
        expansion = 4
        with tf.variable_scope(block_name):
            x = tf.keras.layers.Conv2D(filters, kernel_size=1, use_bias=False)(block_input)
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.keras.activations.relu(x)

            if down_sample:
                x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=2,
                                           padding='same', use_bias=False)(x)
                shortcut = tf.keras.layers.Conv2D(filters*expansion, kernel_size=1,
                                                  strides=2, use_bias=False)(block_input)
            else:
                x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1,
                                           padding='same', use_bias=False)(x)
                shortcut = block_input

            x = tf.layers.batch_normalization(x, training=True)
            x = tf.keras.activations.relu(x)

            x = tf.keras.layers.Conv2D(filters*expansion, kernel_size=1, use_bias=False)(x)
            x = tf.layers.batch_normalization(x, training=True)

            if shortcut.shape[-1] != x.shape[-1]:
                shortcut = tf.keras.layers.Conv2D(filters*expansion, kernel_size=1, strides=1, use_bias=False)(shortcut)
                shortcut = tf.layers.batch_normalization(shortcut, training=True)

            layer = tf.keras.activations.relu(x + shortcut)

        return layer

    def build(self, model_input):
        self.get_residual_layer()

        if self.residual_layer < 50:
            block = self.residual_block
        else:
            block = self.residual_bottleneck

        with tf.variable_scope('conv_1'):
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                       padding='same', use_bias=False)(model_input)
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.keras.activations.relu(x)

        # max pooling layer with kernel 3x3, strides 2
        # x = tf.nn.max_pool(x, ksize=3, strides=2, padding='SAME')

        ########################################################################################################

        for i in range(self.residual_layer_list[0]):
            x = block(x, filters=64, down_sample=False, block_name='conv2_' + str(i))

        ########################################################################################################

        # Down sampling with stride of 2 at the beginning of block
        x = block(x, filters=128, down_sample=True, block_name='conv3_0')

        for i in range(1, self.residual_layer_list[1]):
            x = block(x, filters=128, down_sample=False, block_name='conv3_'+str(i))

        ########################################################################################################

        # Down sampling with stride of 2 at the beginning of block
        x = block(x, filters=256, down_sample=True, block_name='conv4_0')

        for i in range(1, self.residual_layer_list[2]):
            x = block(x, filters=256, down_sample=False, block_name='conv4_'+str(i))

        ########################################################################################################

        # Down sampling with stride of 2 at the beginning of block
        x = block(x, filters=512, down_sample=True, block_name='conv5_0')

        for i in range(1, self.residual_layer_list[3]):
            x = block(x, filters=512, down_sample=False, block_name='conv5_'+str(i))

        ########################################################################################################

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        model = self.fc_layer(x)

        return model

    def get_residual_layer(self):
        if self.residual_layer == 18:
            self.residual_layer_list = [2, 2, 2, 2]
        elif self.residual_layer == 34:
            self.residual_layer_list = [3, 4, 6, 3]
        elif self.residual_layer == 50:
            self.residual_layer_list = [3, 4, 6, 3]
        elif self.residual_layer == 101:
            self.residual_layer_list = [3, 4, 23, 3]
        elif self.residual_layer == 152:
            self.residual_layer_list = [3, 8, 36, 3]
        else:
            raise ValueError('[ResNet] residual layer is invalid, try 18, 34, 50, 101, 152')
