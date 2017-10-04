import tensorflow as tf
from blocks import discriminator_block, conv_block
from layers import conv2d, dense, flatten


class Discriminator:
    def __init__(self, input_shape, normalization=None, is_training=True):
        self.input_shape = input_shape
        self.name = 'model/discriminator'
        self.is_training = is_training
        self.normalization = normalization

    def __call__(self, x, reuse=True, is_feature=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            _x = conv_block(x, filters=32, kernel_size=(4, 4), sampling='down',
                            activation_='lrelu', normalization=self.normalization)

            for i in range(2):
                _x = discriminator_block(_x, filters=32, is_training=self.is_training)
            _x = conv_block(x, filters=64, kernel_size=(4, 4), sampling='down',
                            activation_='lrelu', normalization=self.normalization)

            for i in range(4):
                _x = discriminator_block(_x, filters=64, is_training=self.is_training)
            _x = conv_block(x, filters=128, kernel_size=(4, 4), sampling='down',
                            activation_='lrelu', normalization=self.normalization)

            for i in range(4):
                _x = discriminator_block(_x, filters=128, is_training=self.is_training)
            _x = conv_block(x, filters=256, kernel_size=(4, 4), sampling='down',
                            activation_='lrelu', normalization=self.normalization)

            for i in range(4):
                _x = discriminator_block(_x, filters=256, is_training=self.is_training)
            _x = conv_block(x, filters=512, kernel_size=(4, 4), sampling='down',
                            activation_='lrelu', normalization=self.normalization)

            for i in range(4):
                _x = discriminator_block(_x, filters=512, is_training=self.is_training)
            _x = conv_block(x, filters=1024, kernel_size=(4, 4), sampling='down',
                            activation_='lrelu', normalization=self.normalization)

            if is_feature:
                return _x

            _x = flatten(_x)
            _x = dense(_x, units=1, activation_=None)
            return _x

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]