import tensorflow as tf


class DenseNet:
    def __init__(self, residual_layer, growth_rate=32, num_classes=10):
        self.residual_layer = residual_layer
        self.residual_layer_list = list()
        self.growth_rate = growth_rate
        self.output_classes = num_classes

        self.train_op = None
        self.eval_op = None

    def fc_layer(self, layer_input, scope='fc'):
        with tf.variable_scope(scope):
            layer = tf.layers.flatten(layer_input)
            layer = tf.layers.dense(layer, units=self.output_classes)

        return layer

    def dense_bottleneck(self, layer_input, scope='bottleneck'):
        with tf.variable_scope(scope):
            x = tf.layers.batch_normalization(layer_input, training=True)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=self.growth_rate, kernel_size=1,
                                 strides=1, padding='same', use_bias=False)
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.nn.relu(x)
            layer = tf.layers.conv2d(x, filters=self.growth_rate, kernel_size=3,
                                     strides=1, padding='same', use_bias=False)

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
            block = tf.layers.conv2d(block, filters=self.growth_rate, kernel_size=1,
                                     strides=1, padding='same', use_bias=False)
            block = tf.layers.average_pooling2d(block, pool_size=2, strides=2, padding='same')

        return block

    def build(self, model_input):
        self.get_residual_layer()

        with tf.variable_scope('conv_1'):
            x = tf.layers.conv2d(model_input, filters=self.growth_rate, kernel_size=3,
                                 strides=1, padding='same', use_bias=False)
            # x = tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='same')

            for lidx, lnum in enumerate(self.residual_layer_list):
                x = self.dense_block(x, dn_layers=lnum, scope='dense_blk_'+str(lidx))
                x = self.transition_block(x, scope='trans_blk'+str(lidx))

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