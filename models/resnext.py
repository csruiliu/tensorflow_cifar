import tensorflow as tf


class ResNeXt:
    def __init__(self, cardinality, bottleneck_width, num_classes):
        self.card = cardinality
        self.width = bottleneck_width
        self.output_classes = num_classes

    def res_block(self, block_input, down_sample=False, scope='block'):
        expansion = 2
        group_width = self.card * self.width

        with tf.variable_scope(scope):
            if down_sample:
                strides = 2
                shortcut = tf.keras.layers.Conv2D(filters=expansion*group_width, kernel_size=1,
                                                  strides=2, use_bias=False)(block_input)
                shortcut = tf.layers.batch_normalization(shortcut, training=True)
            else:
                strides = 1
                shortcut = block_input

            x = tf.keras.layers.Conv2D(filters=group_width, kernel_size=1, use_bias=False)(block_input)
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.keras.activations.relu(x)
            '''
            x = tf.keras.layers.Conv2D(filters=group_width, kernel_size=3, strides=strides,
                                       groups=self.card, padding='same', use_bias=False)(x)
            '''

            group_conv_filter = x.shape[-1] // self.card
            output_group_filter = group_width // self.card
            group_conv_list = list()
            for i in range(self.card):
                group_conv_input = x[:, :, :, i*group_conv_filter:(i+1)*group_conv_filter]
                out = tf.keras.layers.Conv2D(filters=output_group_filter, kernel_size=3,
                                             strides=strides, padding='same', use_bias=False)(group_conv_input)
                group_conv_list.append(out)

            x = tf.concat(group_conv_list, axis=3)

            if shortcut.shape[-1] != x.shape[-1]:
                shortcut = tf.keras.layers.Conv2D(group_width, kernel_size=1, strides=1, use_bias=False)(shortcut)
                shortcut = tf.layers.batch_normalization(shortcut, training=True)

            layer = tf.keras.activations.relu(x + shortcut)

        return layer

    def fc_layer(self, layer_input, scope='fc'):
        with tf.variable_scope(scope):
            layer = tf.keras.layers.Flatten()(layer_input)
            layer = tf.keras.layers.Dense(units=self.output_classes)(layer)

        return layer

    def build(self, model_input):
        with tf.variable_scope('conv_1'):
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, use_bias=False)(model_input)
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.keras.activations.relu(x)

        # ======== Stage 1 ======== #
        x = self.res_block(x, scope='blk_1_1')
        x = self.res_block(x, scope='blk_1_2')
        x = self.res_block(x, scope='blk_1_3')

        # Increase bottleneck_width by 2 after each stage.
        self.width = self.width * 2

        # ======== Stage 2 ======== #
        x = self.res_block(x, down_sample=True, scope='blk_2_1')
        x = self.res_block(x, scope='blk_2_2')
        x = self.res_block(x, scope='blk_2_3')

        # Increase bottleneck_width by 2 after each stage.
        self.width = self.width * 2

        # ======== Stage 3 ======== #
        x = self.res_block(x, down_sample=True, scope='blk_3_1')
        x = self.res_block(x, scope='blk_3_2')
        x = self.res_block(x, scope='blk_3_3')

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        model = self.fc_layer(x)

        return model
