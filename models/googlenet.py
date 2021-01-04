import tensorflow as tf


class GoogLeNet:
    def __init__(self, num_classes):
        self.output_classes = num_classes

    @staticmethod
    def inception_block(block_input,
                        n1x1_filters,
                        n3x3red_filters,
                        n3x3_filters,
                        n5x5red_filters,
                        n5x5_filters,
                        pool_filters,
                        scope='inception'):

        with tf.variable_scope(scope):
            # 1x1 conv branch
            x = tf.keras.layers.Conv2D(n1x1_filters, kernel_size=1)(block_input)
            x = tf.layers.batch_normalization(x, training=True)
            ib1 = tf.keras.activations.relu(x)

            # 1x1 conv -> 3x3 conv branch
            x = tf.keras.layers.Conv2D(n3x3red_filters, kernel_size=1)(block_input)
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.keras.activations.relu(x)
            x = tf.keras.layers.Conv2D(n3x3_filters, kernel_size=3, padding='same')(x)
            x = tf.layers.batch_normalization(x, training=True)
            ib2 = tf.keras.activations.relu(x)

            # 1x1 conv -> 5x5 conv branch
            x = tf.keras.layers.Conv2D(n5x5red_filters, kernel_size=1)(block_input)
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.keras.activations.relu(x)
            x = tf.keras.layers.Conv2D(n5x5_filters, kernel_size=5, padding='same')(x)
            x = tf.layers.batch_normalization(x, training=True)
            ib3 = tf.keras.activations.relu(x)

            # 3x3 pool -> 1x1 conv branch
            x = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')(block_input)
            x = tf.keras.layers.Conv2D(pool_filters, kernel_size=1)(x)
            x = tf.layers.batch_normalization(x, training=True)
            ib4 = tf.keras.activations.relu(x)

            layer = tf.concat([ib1, ib2, ib3, ib4], axis=3)

            return layer

    def fc_layer(self, layer_input, scope='fc'):
        with tf.variable_scope(scope):
            layer = tf.keras.layers.Flatten()(layer_input)
            layer = tf.keras.layers.Dense(units=self.output_classes)(layer)

        return layer

    def build(self, model_input):
        with tf.variable_scope('pre_layer'):
            x = tf.keras.layers.Conv2D(filters=192, kernel_size=3, padding='same')(model_input)
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.keras.activations.relu(x)

        x = self.inception_block(x, 64, 96, 128, 16, 32, 32, scope='a3_layer')
        x = self.inception_block(x, 128, 128, 192, 32, 96, 64, scope='b3_layer')

        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        x = self.inception_block(x, 192, 96, 208, 16, 48, 64, scope='a4_layer')
        x = self.inception_block(x, 160, 112, 224, 24, 64, 64, scope='b4_layer')
        x = self.inception_block(x, 128, 128, 256, 24, 64, 64, scope='c4_layer')
        x = self.inception_block(x, 112, 144, 288, 32, 64, 64, scope='d4_layer')
        x = self.inception_block(x, 256, 160, 320, 32, 128, 128, scope='e4_layer')

        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        x = self.inception_block(x, 256, 160, 320, 32, 128, 128, scope='a5_layer')
        x = self.inception_block(x, 384, 192, 384, 48, 128, 128, scope='b5_layer')

        x = tf.keras.layers.AveragePooling2D(pool_size=8, strides=1)(x)

        model = self.fc_layer(x)

        return model
