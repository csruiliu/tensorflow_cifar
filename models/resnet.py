import tensorflow as tf


class ResNet:
    def __init__(self, residual_layer, num_classes):
        self.residual_layer = residual_layer
        self.residual_layer_list = list()
        self.output_classes = num_classes

        self.train_op = None
        self.eval_op = None

    def fc_layer(self, layer_input, scope='fc'):
        with tf.variable_scope(scope):
            layer = tf.layers.flatten(layer_input)
            layer = tf.layers.dense(layer, units=self.output_classes)

        return layer

    # residual block is for resnet-18 and resnet-34
    @staticmethod
    def residual_block(block_input, filters, down_sample=False, block_name='conv'):
        with tf.variable_scope(block_name):
            if down_sample:
                x = tf.layers.conv2d(block_input, filters, 3, strides=2, padding='same', use_bias=False)
                shortcut = tf.layers.conv2d(block_input, filters, 1, strides=2, padding='same', use_bias=False)
                shortcut = tf.layers.batch_normalization(shortcut)
            else:
                x = tf.layers.conv2d(block_input, filters, 3, strides=1, padding='same', use_bias=False)
                shortcut = tf.layers.batch_normalization(block_input)

            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters, 3, strides=1, padding='same', use_bias=False)
            x = tf.layers.batch_normalization(x)
            layer = tf.nn.relu(x + shortcut)

        return layer

    # residual bottleneck is for resnet-50, resnet-101, and resnet-152
    @staticmethod
    def residual_bottleneck(block_input, filters, down_sample=False, block_name='conv'):
        expansion = 4
        with tf.variable_scope(block_name):
            if down_sample:
                x = tf.layers.conv2d(block_input, filters, kernel_size=1,
                                     strides=2, padding='same', use_bias=False)
                shortcut = tf.layers.conv2d(block_input, filters*expansion, kernel_size=1,
                                            strides=2, padding='same', use_bias=False)
            else:
                x = tf.layers.conv2d(block_input, filters, kernel_size=1,
                                     strides=1, padding='same', use_bias=False)
                shortcut = tf.layers.conv2d(block_input, filters*expansion, kernel_size=1,
                                            strides=1, padding='same', use_bias=False)

            shortcut = tf.layers.batch_normalization(shortcut)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters, kernel_size=3, strides=1, padding='same', use_bias=False)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters*expansion, kernel_size=1, strides=1, padding='same', use_bias=False)
            x = tf.layers.batch_normalization(x)
            layer = tf.nn.relu(x + shortcut)

        return layer

    def build(self, model_input):
        if self.residual_layer < 50:
            block = self.residual_block
        else:
            block = self.residual_bottleneck

        self.get_residual_layer()

        with tf.variable_scope('conv_1'):
            x = tf.layers.conv2d(model_input, filters=64, kernel_size=3, strides=1)
            x = tf.layers.batch_normalization(x)

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

    def train(self, model, train_labels, opt, lr):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(train_labels, model)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        reg_loss = tf.losses.get_regularization_loss()
        train_loss = cross_entropy_cost + reg_loss

        if opt == 'Adam':
            self.train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
        elif opt == 'SGD':
            self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(train_loss)
        elif opt == 'Adagrad':
            self.train_op = tf.train.AdagradOptimizer(lr).minimize(train_loss)
        elif opt == 'Momentum':
            self.train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(train_loss)
        else:
            raise ValueError('Optimizer is not recognized')

        return self.train_op

    def evaluate(self, model, eval_labels):
        prediction = tf.equal(tf.argmax(model, -1), tf.argmax(eval_labels, -1))
        self.eval_op = tf.reduce_mean(tf.cast(prediction, tf.float32))

        return self.eval_op

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
            raise ValueError('[ResNet] residual layer is invalid')