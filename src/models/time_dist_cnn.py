seed = 10707
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)

from models.base_model import VQANet
from keras.layers import *
from keras import Model


class TimeDistributedCNNNet(VQANet):
    def __init__(self, lstm_dim, n_answers, model_path, log_path, VOCAB_SIZE, MAX_QUESTION_LEN, question_embed_dim=None):
        super().__init__(question_embed_dim=question_embed_dim,
                         lstm_dim=lstm_dim,
                         n_answers=n_answers,
                         model_path=model_path,
                         log_path=log_path,
                         VOCAB_SIZE=VOCAB_SIZE,
                         MAX_QUESTION_LEN=MAX_QUESTION_LEN)
        self.build()

    def build(self):
        # with tf.device('/cpu:0'):
        image_input = Input(shape=(4096,),
                               dtype='float32',
                               name='image_input')
        image_embedding = Dense(units=self.question_embed_dim,
                                activation='relu',
                                name='image_embedding')(inputs=image_input)
        image_embedding = RepeatVector(n=self.MAX_QUESTION_LEN)(inputs=image_embedding)


        question_input = Input(shape=(self.MAX_QUESTION_LEN,),
                               dtype='int32',
                               name='question_input')
        question_input_masked = Masking(mask_value=0,
                                        input_shape=(self.MAX_QUESTION_LEN,),
                                name='question_input')(inputs=question_input)
        question_embedding = Embedding(input_dim=self.VOCAB_SIZE,
                                       output_dim=self.question_embed_dim,
                                       input_length=self.MAX_QUESTION_LEN,
                                       name='question_embedding')(inputs=question_input_masked)


        image_question_embedding = Concatenate(axis=-1,
                                               name='image_question_embedding')(inputs=[image_embedding, question_embedding])
        question_embedding, last_h, _ = LSTM(units=self.lstm_dim,
                                             return_sequences=True,
                                             return_state=True,
                                             name='question_lstm')(inputs=image_question_embedding)
        question_pred = TimeDistributed(layer=Dense(units=self.VOCAB_SIZE, activation='softmax'),
                                        name='question_classifier')(inputs=question_embedding)


        answer_fc_1 = Dense(units=1000,
                            activation='relu',
                            name='answer_fc_1')(inputs=last_h)
        answer_fc_2 = Dense(units=1000,
                            activation='relu',
                            name='answer_fc_2')(inputs=answer_fc_1)
        answer_pred = Dense(units=self.n_answers,
                            activation='softmax',
                            name='answer_classifier')(inputs=answer_fc_2)


        self.model = Model(inputs=[image_input, question_input],
                           outputs=[question_pred, answer_pred])
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
