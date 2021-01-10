import tensorflow as tf


class ShuffleNetV2:
    def __init__(self,  complexity=2, num_classes=10):
        self.output_classes = num_classes
        self.complexity = complexity
        if self.complexity == 0.5:
            self.out_filters = [48, 96, 192]
        elif self.complexity == 1:
            self.out_filters = [116, 232, 464]
        elif self.complexity == 1.5:
            self.out_filters = [176, 352, 704]
        elif self.complexity == 2:
            self.out_filters = [244, 488, 976]
        else:
            raise ValueError('[ShuffleNetV2] complexity is invalid, try 0.5, 1, 1.5, 2')

    @staticmethod
    def split_unit(unit_input, split_ratio=0.5):
        input_chn = int(unit_input.shape[-1])
        chn_split_x = input_chn * split_ratio
        chn_split_y = input_chn * (1 - split_ratio)
        return unit_input[:, :, :, 0:int(chn_split_x)], unit_input[:, :, :, int(chn_split_y):input_chn]

    @staticmethod
    def shuffle_unit(unit_input, groups=2):
        img_num, img_height, img_width, img_chn = unit_input.shape
        x = tf.reshape(unit_input, [-1, img_height, img_width, groups, img_chn//groups])
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
        unit = tf.reshape(x, [-1, img_height, img_width, img_chn])

        return unit

    def bottleneck(self, block_input, out_filters, chn_split_ratio=0.5, down_sample=False, scope='blk'):
        with tf.variable_scope(scope):
            if down_sample:
                blk_strides = 2
                mid_filters = out_filters // 2
                x_right = block_input
                x_left = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2,
                                                         padding='same', use_bias=False)(block_input)
                x_left = tf.layers.batch_normalization(x_left, training=True)

                x_left = tf.keras.layers.Conv2D(filters=mid_filters, kernel_size=1,
                                                strides=1, use_bias=False)(x_left)
                x_left = tf.layers.batch_normalization(x_left, training=True)
                x_left = tf.keras.activations.relu(x_left)
            else:
                blk_strides = 1
                mid_filters = int(int(block_input.shape[-1]) * chn_split_ratio)
                x_left, x_right = self.split_unit(block_input)

            x_right = tf.keras.layers.Conv2D(filters=mid_filters, kernel_size=1, strides=1,
                                             padding='same', use_bias=False)(x_right)
            x_right = tf.layers.batch_normalization(x_right, training=True)
            x_right = tf.keras.activations.relu(x_right)

            x_right = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=blk_strides,
                                                      padding='same', use_bias=False)(x_right)
            x_right = tf.layers.batch_normalization(x_right, training=True)

            x_right = tf.keras.layers.Conv2D(filters=mid_filters, kernel_size=1, strides=1,
                                             padding='same', use_bias=False)(x_right)
            x_right = tf.layers.batch_normalization(x_right, training=True)
            x_right = tf.keras.activations.relu(x_right)

            x = tf.concat([x_left, x_right], axis=3)
            block = self.shuffle_unit(x)

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

        with tf.variable_scope('stage_2'):
            x = self.bottleneck(x, out_filters=self.out_filters[0], chn_split_ratio=0.5,
                                down_sample=True, scope='blk_0')
            for i in range(1, 4):
                x = self.bottleneck(x, out_filters=self.out_filters[0], chn_split_ratio=0.5,
                                    down_sample=False, scope='blk_'+str(i))

        with tf.variable_scope('stage_3'):
            x = self.bottleneck(x, out_filters=self.out_filters[1], chn_split_ratio=0.5,
                                down_sample=True, scope='blk_0')
            for i in range(1, 8):
                x = self.bottleneck(x, out_filters=self.out_filters[1], chn_split_ratio=0.5,
                                    down_sample=False, scope='blk_'+str(i))

        with tf.variable_scope('stage_4'):
            x = self.bottleneck(x, out_filters=self.out_filters[2], chn_split_ratio=0.5,
                                down_sample=True, scope='blk_0')
            for i in range(1, 4):
                x = self.bottleneck(x, out_filters=self.out_filters[2], chn_split_ratio=0.5,
                                    down_sample=False, scope='blk_'+str(i))

        x = tf.keras.layers.Conv2D(filters=1024, kernel_size=1, strides=1)(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        model = self.fc_layer(x, scope='fc')

        return model
