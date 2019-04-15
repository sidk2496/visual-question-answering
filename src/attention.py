import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K


class SpatialAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        img_feat_shape, word_embed_shape = input_shape
        self.batch_size, self.depth, self.height, self.width = img_feat_shape
        self.embed_dim = word_embed_shape[1]

        self.A = self.add_weight(name='attention',
                                 shape=(self.depth, self.embed_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super().build(input_shape=input_shape)

    def call(self, inputs, **kwargs):
        img_feat, word_embed = inputs
        img_feat = K.permute_dimensions(img_feat, pattern=(0, 2, 3, 1))
        attention_weights = K.dot(K.dot(img_feat, self.A))
        attention_weights = K.reshape(attention_weights, shape=(self.batch_size, self.height * self.width,
                                                                self.embed_dim))
        attention_weights = K.batch_dot(attention_weights, word_embed, axes=[2, 1])
        attention_weights = K.softmax(attention_weights, axis=-1)
        attention_weights = K.reshape(attention_weights, shape=(self.batch_size, self.height, self.width, 1))
        attended_img_feat = tf.math.multiply(img_feat, attention_weights)
        return K.permute_dimensions(attended_img_feat, pattern=(0, 3, 1, 2))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        return config
