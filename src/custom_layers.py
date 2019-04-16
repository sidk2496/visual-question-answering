import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import backend as K


class SpatialAttention(Layer):
    def __init__(self, weights_name, **kwargs):
        super().__init__(**kwargs)
        self.weights_name = weights_name

    def build(self, input_shape):
        img_feat_shape, word_embed_shape = input_shape
        self.batch_size, self.timesteps, self.depth, self.height, self.width = img_feat_shape
        self.embed_dim = word_embed_shape[2]
        # A -> d_i,d_e
        self.A = self.add_weight(name=self.weights_name,
                                 shape=(self.depth, self.embed_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super().build(input_shape=input_shape)

    def call(self, inputs, **kwargs):
        img_feat, word_embed = inputs
        # b,t,d_i,h,w -> b,t,h,w,d_i
        img_feat = K.permute_dimensions(img_feat, pattern=(0, 1, 3, 4, 2))
        # b,t,h,w,d_i -> b,t,h,w,d_e
        attention_weights = K.dot(img_feat, self.A)
        # b,t,h,w,d_e -> b,t,h*w,d_e
        attention_weights = K.reshape(attention_weights, shape=(self.batch_size, self.timesteps,
                                                                self.height * self.width, self.embed_dim))
        # b,t,d_e -> b,t,1,d_e
        word_embed = K.reshape(word_embed, shape=(self.batch_size, self.timesteps, 1, self.embed_dim))
        # b,t,h*w,d_e -> b,t,h*w,1
        attention_weights = K.batch_dot(attention_weights, word_embed, axes=[3, 3])
        attention_weights = K.softmax(attention_weights, axis=-2)
        # b,t,h*w,1 -> b,t,h,w,1
        attention_weights = K.reshape(attention_weights, shape=(self.batch_size, self.height, self.width, 1))
        attended_img_feat = tf.math.multiply(img_feat, attention_weights)
        # b,t,h,w,d_i -> b,t,d_i,h,w
        return K.permute_dimensions(attended_img_feat, pattern=(0, 1, 4, 2, 3))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        return config



class ConvAttention(Layer):
    def __init__(self, cnn):
        self.cnn = cnn
        super().__init__()

    def build(self, input_shape):
        img_shape, word_embed_shape = input_shape
        self.batch_size, self.depth, self.height, self.width = img_shape
        self.timesteps, self.embed_dim = word_embed_shape[1:]
        self.conv_layers = self.layers[17:21]
        for layer in self.conv_layers:
            self._trainable_weights += layer._trainable_weights
        self.new_conv_1 = Conv2D(filters=128,
                                 kernel_size=(3, 3),
                                 activation='relu',
                                 padding='same',
                                 data_format='channels_first')
        self.new_conv_2 = Conv2D
        self.
        super().build(input_shape=input_shape)


    def call(self, inputs, **kwargs):
        img_feat, word_embed = inputs
        self.cnn.

    def compute_output_shape(self, input_shape):

    def get_config(self):
        base_config = super().get_config()
        return base_config

