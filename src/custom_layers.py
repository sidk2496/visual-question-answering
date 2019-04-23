import tensorflow as tf
from keras.layers import *
from keras import backend as K


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
        attention_weights = K.reshape(attention_weights, shape=(-1, self.timesteps,
                                                                self.height * self.width, self.embed_dim))
        # b,t,d_e -> b,t,1,d_e
        word_embed = K.reshape(word_embed, shape=(-1, self.timesteps, 1, self.embed_dim))
        # b,t,h*w,d_e -> b,t,h*w,1
        attention_weights = K.batch_dot(attention_weights, word_embed, axes=[3, 3])
        attention_weights = K.softmax(attention_weights, axis=-2)
        # b,t,h*w,1 -> b,t,h,w,1
        attention_weights = K.reshape(attention_weights, shape=(-1, self.height, self.width, 1))
        attended_img_feat = tf.math.multiply(img_feat, attention_weights)
        # b,t,h,w,d_i -> b,t,d_i,h,w
        return K.permute_dimensions(attended_img_feat, pattern=(0, 1, 4, 2, 3))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        return config



class Context(Layer):
    def __init__(self, cnn):
        self.cnn = cnn
        super().__init__()

    def build(self, input_shape):

        img_shape, word_embed_shape = input_shape
        self.batch_size, self.depth, self.height, self.width = img_shape
        self.timesteps, self.embed_dim = word_embed_shape[1:]

        # add vgg19 conv layers
        self.conv_layers = [TimeDistributed(layer=layer, name='vgg_conv_{}'.format(i + 1))
                            for i, layer in enumerate(self.cnn.layers[17:21])]

        # add new conv layers
        self.conv_layers += [TimeDistributed(Conv2D(filters=512,
                                                    kernel_size=(3, 3),
                                                    activation='relu',
                                                    padding='same',
                                                    data_format='channels_first'),
                                             name='new_conv_{}'.format(i)) for i in range(1, 3)]

        # add attention layers
        self.attention_layers = [SpatialAttention(weights_name='attention_{}'.format(i))
                                 for i in range(1, 7)]

        # add max-pool layers
        pool_sizes = [(2, 2)] * 2 + [(3, 3)]
        self.pool_layers = [TimeDistributed(layer=MaxPool2D(pool_size=pool_size), name='pool_{}'.format(i + 1))
                            for i, pool_size in enumerate(pool_sizes)]

        # build layers
        input_shapes = [(self.batch_size, self.timesteps, 512, 7, 7),
                        (self.batch_size, self.timesteps, 512, 3, 3)] + \
                       [((self.batch_size, self.timesteps, 512, 14, 14), (self.batch_size, self.timesteps, 1024))] * 4 + \
                       [((self.batch_size, self.timesteps, 512, 7, 7), (self.batch_size, self.timesteps, 1024)),
                        ((self.batch_size, self.timesteps, 512, 3, 3), (self.batch_size, self.timesteps, 1024)),
                        (self.batch_size, self.timesteps, 512, 14, 14),
                        (self.batch_size, self.timesteps, 512, 7, 7),
                        (self.batch_size, self.timesteps, 512, 3, 3)]

        for (layer, input_shape) in zip(self.conv_layers[-2:] + self.attention_layers + self.pool_layers,
                                        input_shapes):
            layer.build(input_shape=input_shape)

        for layer in self.conv_layers + self.attention_layers + self.pool_layers:
            self._trainable_weights += layer._trainable_weights

        super().build(input_shape=input_shape)

    def call(self, inputs, **kwargs):
        img_feat, word_embed = inputs
        img_feat = K.reshape(img_feat, shape=(-1, 1, self.depth, self.height, self.width))
        img_feat = K.repeat_elements(img_feat, rep=self.timesteps, axis=1)

        context = img_feat
        for i in range(3):
            context = self.conv_layers[i](context)
            print(context)
            context = self.attention_layers[i]([context, word_embed])
            print(context)

        for i in range(3):
            context = self.conv_layers[i + 3](context)
            print(context)
            context = self.attention_layers[i + 3]([context, word_embed])
            print(context)
            context = self.pool_layers[i](context)
            print(context)
        
        return K.reshape(context, shape=(-1, self.timesteps, 512))

    def compute_output_shape(self, input_shape):
        return self.batch_size, self.timesteps, 512

    def get_config(self):
        base_config = super().get_config()
        return base_config

