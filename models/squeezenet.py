import tensorflow as tf


class SqueezeNet:
    def __init__(self, num_classes=10):
        self.output_classes = num_classes

    @staticmethod
    def fire_block(block_input, filters_s1x1, filters_e1x1, filters_e3x3, scope='fire_blk'):
        with tf.variable_scope(scope):
            x = tf.keras.layers.Conv2D(filters=filters_s1x1, kernel_size=1, padding='same')(block_input)
            bn_x = tf.layers.batch_normalization(x, training=True)
            squeeze_x = tf.keras.activations.relu(bn_x)

            expand_x1 = tf.keras.layers.Conv2D(filters=filters_e1x1, kernel_size=1, padding='same')(squeeze_x)
            expand_x1 = tf.layers.batch_normalization(expand_x1, training=True)
            expand_x1 = tf.keras.activations.relu(expand_x1)

            expand_x3 = tf.keras.layers.Conv2D(filters=filters_e3x3, kernel_size=3, padding='same')(squeeze_x)
            expand_x3 = tf.layers.batch_normalization(expand_x3, training=True)
            expand_x3 = tf.keras.activations.relu(expand_x3)

            layer = tf.concat([expand_x1, expand_x3], axis=3)

            return layer

    def fc_layer(self, layer_input, scope='fc'):
        with tf.variable_scope(scope):
            layer = tf.keras.layers.Flatten()(layer_input)
            layer = tf.keras.layers.Dense(units=self.output_classes)(layer)

        return layer

    def build(self, model_input):
        x = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=2, padding='same')(model_input)
        x = tf.layers.batch_normalization(x, training=True)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
        x = self.fire_block(x, filters_s1x1=16, filters_e1x1=64, filters_e3x3=64, scope='fire_blk_2')
        x = self.fire_block(x, filters_s1x1=16, filters_e1x1=64, filters_e3x3=64, scope='fire_blk_3')
        x = self.fire_block(x, filters_s1x1=32, filters_e1x1=128, filters_e3x3=128, scope='fire_blk_4')
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
        x = self.fire_block(x, filters_s1x1=32, filters_e1x1=128, filters_e3x3=128, scope='fire_blk_5')
        x = self.fire_block(x, filters_s1x1=48, filters_e1x1=192, filters_e3x3=192, scope='fire_blk_6')
        x = self.fire_block(x, filters_s1x1=48, filters_e1x1=192, filters_e3x3=192, scope='fire_blk_7')
        x = self.fire_block(x, filters_s1x1=64, filters_e1x1=256, filters_e3x3=256, scope='fire_blk_8')
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
        x = self.fire_block(x, filters_s1x1=64, filters_e1x1=256, filters_e3x3=256, scope='fire_blk_9')
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Conv2D(filters=10, kernel_size=1, padding='same')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        model = self.fc_layer(x, scope='fc')

        return model
