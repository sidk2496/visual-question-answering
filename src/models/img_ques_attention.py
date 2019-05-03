seed = 10707
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)

from models.base_model import VQANet, dummy, custom_acc
from keras.layers import *
from keras import backend as K
from keras import Model
from keras.activations import softmax
from keras.regularizers import l2


class ImgQuesAttentionNet(VQANet):
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
        image_input = Input(shape=(512, 7, 7),
                            dtype='float32',
                            name='image_input')
        image_feat = Reshape(target_shape=(512, 49))(inputs=image_input)
        image_feat = Permute(dims=(2, 1))(inputs=image_feat)


        question_input = Input(shape=(self.MAX_QUESTION_LEN,),
                               dtype='int32',
                               name='question_input')
        question_input_masked = Masking(mask_value=0,
                                 input_shape=(self.MAX_QUESTION_LEN,))(inputs=question_input)
        question_embedding = Embedding(input_dim=self.VOCAB_SIZE,
                                       output_dim=self.question_embed_dim,
                                       input_length=self.MAX_QUESTION_LEN,
                                       name='question_embedding')(inputs=question_input_masked)


        question_embedding = Bidirectional(layer=LSTM(units=self.lstm_dim,
                                                      return_sequences=True),
                                           name='question_lstm_1')(inputs=question_embedding)
        question_embedding = Bidirectional(layer=LSTM(units=self.lstm_dim,
                                                      return_sequences=True),
                                           name='question_lstm_2')(inputs=question_embedding)


        vq_attention_weights = TimeDistributed(layer=Dense(units=2 * self.lstm_dim))(inputs=image_feat)
        vq_attention_weights = Lambda(lambda x: K.batch_dot(*x, axes=[2, 2]))(inputs=[question_embedding,
                                                                                      vq_attention_weights])
        vq_attention_weights = Lambda(lambda x: softmax(x, axis=-1),
                                      name='vq_attention_weights')(inputs=vq_attention_weights)
        vq_context = Lambda(lambda x: K.batch_dot(*x, axes=[2, 1]),
                            name='vq_context')(inputs=[vq_attention_weights, image_feat])
        vq_context_question_embedding = Concatenate(axis=-1)(inputs=[vq_context, question_embedding])


        question_pred = TimeDistributed(layer=Dense(units=self.VOCAB_SIZE,
                                                    activation='softmax'),
                                        name='question_classifier')(inputs=vq_context_question_embedding)


        qa_attention_weights = TimeDistributed(layer=Dense(units=1000,
                                                           activation='relu'))(inputs=vq_context_question_embedding)
        # qa_attention_weights = Dropout(rate=0.5, seed=seed)(inputs=qa_attention_weights)
        qa_attention_weights = TimeDistributed(layer=Dense(units=256,
                                                           activation='relu'))(inputs=qa_attention_weights)
        # qa_attention_weights = Dropout(rate=0.5, seed=seed)(inputs=qa_attention_weights)
        qa_attention_weights = TimeDistributed(layer=Dense(units=1))(inputs=qa_attention_weights)
        qa_attention_weights = Lambda(lambda x: softmax(x, axis=1),
                                      name='qa_attention_weights')(inputs=qa_attention_weights)
        qa_context = Dot(axes=1)(inputs=[qa_attention_weights, vq_context_question_embedding])
        qa_context = Reshape(target_shape=(2 * self.lstm_dim + 512,))(inputs=qa_context)


        answer_fc_1 = Dense(units=1000,
                            activation='relu',
                            name='answer_fc_1')(inputs=qa_context)
        # answer_fc_1 = Dropout(rate=0.5, seed=seed)(inputs=answer_fc_1)
        answer_fc_2 = Dense(units=1000,
                            activation='relu',
                            name='answer_fc_2')(inputs=answer_fc_1)
        # answer_fc_2 = Dropout(rate=0.5, seed=seed)(inputs=answer_fc_2)
        answer_pred = Dense(units=self.n_answers,
                            activation='softmax',
                            name='answer_classifier')(inputs=answer_fc_2)


        best_ans = Lambda(lambda x: K.argmax(x, axis=-1))(inputs=answer_pred)
        best_ans = Reshape(target_shape=(1,))(inputs=best_ans)
        best_ans = RepeatVector(n=10, )(inputs=best_ans)
        best_ans = Reshape(target_shape=(10,),
                           name='best_ans')(inputs=best_ans)


        self.model = Model(inputs=[image_input, question_input],
                           outputs=[question_pred, answer_pred, best_ans])
        losses = {
            'question_classifier': 'categorical_crossentropy',
            'answer_classifier': 'categorical_crossentropy',
            'best_ans': dummy
        }
        metrics = {
            'question_classifier': 'acc',
            'answer_classifier': 'acc',
            'best_ans': custom_acc,
        }
        self.model.compile(loss=losses,
                           optimizer='adam',
                           metrics=metrics)

        print(self.model.metrics_names)