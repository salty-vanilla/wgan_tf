import tensorflow as tf
from layers import dense, reshape, batch_norm, activation, conv2d
from blocks import residual_block, conv_block


class Generator:
    def __init__(self, noise_dim, last_activation='tanh',
                 color_mode='rgb', normalize='batch', upsampling='subpixel', is_training=True):
        self.noise_dim = noise_dim
        self.last_actovation = last_activation
        self.name = 'model/generator'
        assert color_mode in ['grayscale', 'gray', 'rgb']
        self.channel = 1 if color_mode in ['grayscale', 'gray'] else 3
        self.normalize = normalize
        self.upsampling = upsampling
        self.is_training = is_training

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            _x = dense(x, 16 * 16 * 1024, activation_=None)
            _x = reshape(_x, (16, 16, 1024))
            _x = batch_norm(_x, is_training=self.is_training)
            _x = activation(_x, 'relu')

            _x = conv2d(_x, filters=64, kernel_size=(1, 1), activation_='relu')
            residual_inputs = _x

            with tf.name_scope('residual_blocks'):
                for i in range(16):
                    _x = residual_block(_x, filters=64, activation_='relu', is_training=self.is_training,
                                        sampling='same', normalization='batch', dropout=0., mode='conv_first')

            _x = conv_block(_x, filters=64, activation_='relu', is_training=self.is_training,
                            sampling='same', normalization='batch', dropout=0., mode='conv_first')
            _x += residual_inputs

            with tf.name_scope('upsampling_blocks'):
                for i in range(3):
                    _x = conv_block(_x, filters=64, activation_='relu', is_training=self.is_training,
                                    sampling=self.upsampling, normalization='batch', dropout=0., mode='conv_first')

            _x = conv_block(_x, filters=self.channel, activation_=self.last_actovation, is_training=self.is_training,
                            sampling='same', normalization=None, dropout=0., mode='conv_first')
            return _x

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]
