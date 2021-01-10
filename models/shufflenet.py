import tensorflow as tf


class ShuffleNet:
    def __init__(self, num_groups=2, num_classes=10):
        self.output_classes = num_classes
        self.groups = num_groups
        if num_groups == 2:
            self.out_filters = [200, 400, 800]
        elif num_groups == 3:
            self.out_filters = [240, 480, 960]
        elif num_groups == 4:
            self.out_filters = [272, 544, 1088]
        elif num_groups == 8:
            self.out_filters = [384, 768, 1536]
        else:
            raise ValueError('[ShuffleNet] number of residual layer is invalid, try 121, 169, 201, 264')

    @staticmethod
    def group_conv(layer_input,
                   filters,
                   kernel_size,
                   strides,
                   padding,
                   groups,
                   use_bias):

        input_group_filters = layer_input.shape[-1] // groups
        output_group_filters = filters // groups

        group_conv_list = list()
        for i in range(groups):
            group_conv_input = layer_input[:, :, :, i * input_group_filters:(i + 1) * input_group_filters]
            out = tf.keras.layers.Conv2D(filters=output_group_filters, kernel_size=kernel_size,
                                         strides=strides, padding=padding, use_bias=use_bias)(group_conv_input)
            group_conv_list.append(out)

        layer = tf.concat(group_conv_list, axis=3)

        return layer

    @staticmethod
    def shuffle_unit(unit_input, groups):
        img_num, img_height, img_width, img_chn = unit_input.shape
        x = tf.reshape(unit_input, [-1, img_height, img_width, groups, img_chn//groups])
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
        unit = tf.reshape(x, [-1, img_height, img_width, img_chn])

        return unit

    def bottleneck(self, block_input, out_filters, use_groups=False, down_sample=False, scope='blk'):
        with tf.variable_scope(scope):
            blk_strides = 1
            mid_filters = out_filters // 4

            if down_sample:
                blk_strides = 2
                shortcut = tf.keras.layers.AveragePooling2D(pool_size=3, strides=2, padding='same')(block_input)
            else:
                shortcut = block_input

            if use_groups:
                x = self.group_conv(block_input, filters=mid_filters, kernel_size=1, strides=1,
                                    padding='valid', groups=self.groups, use_bias=False)
                x = tf.layers.batch_normalization(x, training=True)
                x = tf.keras.activations.relu(x)

                x = self.shuffle_unit(x, self.groups)

                x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=blk_strides,
                                                    padding='same', use_bias=False)(x)
                x = tf.layers.batch_normalization(x, training=True)

                x = self.group_conv(x, filters=out_filters, kernel_size=1, strides=1,
                                    padding='valid', groups=self.groups, use_bias=False)
                x = tf.layers.batch_normalization(x, training=True)

            else:
                x = tf.keras.layers.Conv2D(filters=mid_filters, kernel_size=1, use_bias=False)(block_input)
                x = tf.layers.batch_normalization(x, training=True)
                x = tf.keras.activations.relu(x)

                x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=blk_strides,
                                                    padding='same', use_bias=False)(x)
                x = tf.layers.batch_normalization(x, training=True)
                x = tf.keras.activations.relu(x)

                x = tf.keras.layers.Conv2D(filters=out_filters, kernel_size=1, use_bias=False)(x)
                x = tf.layers.batch_normalization(x, training=True)

            if down_sample:
                x = tf.concat([x, shortcut], axis=3)
            else:
                if shortcut.shape[-1] != x.shape[-1]:
                    shortcut = tf.keras.layers.Conv2D(out_filters, kernel_size=1, strides=1, use_bias=False)(shortcut)
                    shortcut = tf.layers.batch_normalization(shortcut, training=True)
                x = x + shortcut

            block = tf.keras.activations.relu(x)

        return block

    def fc_layer(self, layer_input, scope='fc'):
        with tf.variable_scope(scope):
            layer = tf.keras.layers.Flatten()(layer_input)
            layer = tf.keras.layers.Dense(units=self.output_classes)(layer)

        return layer

    def build(self, model_input):
        with tf.variable_scope('stage_1'):
            x = tf.keras.layers.Conv2D(filters=24, kernel_size=3, strides=2, use_bias=False)(model_input)
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.keras.activations.relu(x)
            # x = tf.keras.layers.MaxPool2D(pool_size=3, stride=2, padding='same')(x)

        with tf.variable_scope('stage_2'):
            x = self.bottleneck(x, out_filters=self.out_filters[0], use_groups=False, down_sample=True, scope='blk_0')
            for i in range(1, 4):
                x = self.bottleneck(x, out_filters=self.out_filters[0], use_groups=False,
                                    down_sample=False, scope='blk_'+str(i))

        with tf.variable_scope('stage_3'):
            x = self.bottleneck(x, out_filters=self.out_filters[1], use_groups=True, down_sample=True, scope='blk_0')
            for i in range(1, 8):
                x = self.bottleneck(x, out_filters=self.out_filters[1], use_groups=True,
                                    down_sample=False, scope='blk_'+str(i))

        with tf.variable_scope('stage_4'):
            x = self.bottleneck(x, out_filters=self.out_filters[2], use_groups=True, down_sample=True, scope='blk_0')
            for i in range(1, 4):
                x = self.bottleneck(x, out_filters=self.out_filters[2], use_groups=True,
                                    down_sample=False, scope='blk_'+str(i))

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        model = self.fc_layer(x, scope='fc')

        return model
