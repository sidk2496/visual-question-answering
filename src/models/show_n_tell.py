seed = 10707
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)

from models.base_model import VQANet, dummy, custom_acc
from keras.layers import *
from keras import backend as K
from keras import Model
from keras.regularizers import l2


class ShowNTellNet(VQANet):
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
        image_embedding_lstm_input = Reshape(target_shape=(1, self.question_embed_dim))(inputs=image_embedding)


        question_input = Input(shape=(self.MAX_QUESTION_LEN,),
                               dtype='int32',
                               name='question_input')
        question_input_masked = Masking(mask_value=0,
                                        input_shape=(self.MAX_QUESTION_LEN,))(inputs=question_input)
        question_embedding = Embedding(input_dim=self.VOCAB_SIZE,
                                       output_dim=self.question_embed_dim,
                                       input_length=self.MAX_QUESTION_LEN,
                                       name='question_embedding')(inputs=question_input_masked)


        image_question_embedding = Concatenate(axis=1,
                                               name='image_question_embedding')(inputs=[image_embedding_lstm_input,
                                                                                        question_embedding])
        question_embedding = Bidirectional(layer=LSTM(units=self.lstm_dim,
                                                      return_sequences=True,
                                                      kernel_regularizer=l2(0.001)),
                                           name='question_lstm_1')(inputs=image_question_embedding)
        question_embedding, last_fh, _, last_bh, _ = Bidirectional(layer=LSTM(units=self.lstm_dim,
                                                                              return_sequences=True,
                                                                              return_state=True,
                                                                              kernel_regularizer=l2(0.001)),
                                                                   name='question_lstm_2')(inputs=question_embedding)
        question_pred = TimeDistributed(layer=Dense(units=self.VOCAB_SIZE,
                                                    activation='softmax'))(inputs=question_embedding)
        question_pred = Lambda(lambda x: x[:, 1:, :],
                               name='question_classifier')(inputs=question_pred)


        answer_fc_input = Concatenate(axis=-1,
                                      name='answer_fc_input')(inputs=[image_embedding,
                                                                      last_fh, last_bh])
        answer_fc_1 = Dense(units=1000,
                            activation='relu',
                            name='answer_fc_1')(inputs=answer_fc_input)
        answer_fc_2 = Dense(units=1000,
                            activation='relu',
                            name='answer_fc_2')(inputs=answer_fc_1)
        answer_pred = Dense(units=self.n_answers,
                            activation='softmax',
                            name='answer_classifier')(inputs=answer_fc_2)


        best_ans = Lambda(lambda x: K.argmax(x, axis=-1))(inputs=answer_pred)
        best_ans = Reshape(target_shape=(1,))(inputs=best_ans)
        best_ans = RepeatVector(n=10)(inputs=best_ans)
        best_ans = Reshape(target_shape=(10,), name='best_ans')(inputs=best_ans)


        self.model = Model(inputs=[image_input, question_input],
                           outputs=[answer_pred, best_ans])
        losses = {
            # 'question_classifier': 'categorical_crossentropy',
            'answer_classifier': 'categorical_crossentropy',
            'best_ans': dummy
        }
        metrics = {
            # 'question_classifier': 'acc',
            'answer_classifier': 'acc',
            'best_ans': custom_acc,
        }
        self.model.compile(loss=losses,
                           optimizer='adam',
                           metrics=metrics)