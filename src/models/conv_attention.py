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
from keras.activations import softmax
from custom_layers import Context


class ConvAttentionNet(VQANet):
    def __init__(self, lstm_dim, n_answers, model_path, log_path, VOCAB_SIZE, MAX_QUESTION_LEN,
                 question_embed_dim):
        super().__init__(question_embed_dim=question_embed_dim,
                         lstm_dim=lstm_dim,
                         n_answers=n_answers,
                         model_path=model_path,
                         log_path=log_path,
                         VOCAB_SIZE=VOCAB_SIZE,
                         MAX_QUESTION_LEN=MAX_QUESTION_LEN)
        self.build()

    def build(self):
        # input
        image_input = Input(shape=(3, 224, 224))

        vgg_layer_16 = image_input
        for i in range(1, 17):
            vgg_layer_16 = self.cnn.layers[i](vgg_layer_16)
            self.cnn.layers[i].trainable = False

        question_input = Input(shape=(self.MAX_QUESTION_LEN,),
                               dtype='int32',
                               name='question_input')
        question_input_masked = Masking(mask_value=0,
                                        input_shape=(self.MAX_QUESTION_LEN,))(inputs=question_input)


        # lstm question embeddings
        question_embedding = Embedding(input_dim=self.VOCAB_SIZE,
                                       output_dim=self.question_embed_dim,
                                       input_length=self.MAX_QUESTION_LEN,
                                       name='question_embedding')(inputs=question_input_masked)
        question_embedding = Bidirectional(layer=LSTM(units=self.lstm_dim,
                                                      return_sequences=True,
                                                      kernel_regularizer=l2(0)),
                                           name='question_lstm_1')(inputs=question_embedding)
        question_embedding = Bidirectional(layer=LSTM(units=self.lstm_dim,
                                                      return_sequences=True,
                                                      kernel_regularizer=l2(0)),
                                           name='question_lstm_2')(inputs=question_embedding)


        print(question_embedding)

        # attended CNN features
        vq_context = Context(cnn=self.cnn)(inputs=[vgg_layer_16, question_embedding])
        vq_context_question_embedding = Concatenate(axis=-1)(inputs=[vq_context, question_embedding])

        # question_pred = TimeDistributed(layer=Dense(units=self.VOCAB_SIZE,
        #                                             activation='softmax',
        #                                             kernel_regularizer=l2(0)),
        #                                 name='question_classifier')(inputs=vq_context_question_embedding)

        # question-answer attention
        qa_attention_weights = TimeDistributed(layer=Dense(units=1000,
                                                           activation='relu',
                                                           kernel_regularizer=l2(0)))(inputs=vq_context_question_embedding)
        # qa_attention_weights = Dropout(rate=0.5, seed=seed)(inputs=qa_attention_weights)
        qa_attention_weights = TimeDistributed(layer=Dense(units=256,
                                                           activation='relu',
                                                           kernel_regularizer=l2(0)))(inputs=qa_attention_weights)
        # qa_attention_weights = Dropout(rate=0.5, seed=seed)(inputs=qa_attention_weights)
        qa_attention_weights = TimeDistributed(layer=Dense(units=1,
                                                           kernel_regularizer=l2(0)))(inputs=qa_attention_weights)
        qa_attention_weights = Lambda(lambda x: softmax(x, axis=1),
                                      name='qa_attention_weights')(inputs=qa_attention_weights)
        qa_context = Dot(axes=1)(inputs=[qa_attention_weights, vq_context_question_embedding])

        print(qa_context)
        qa_context = Reshape(target_shape=(2 * self.lstm_dim + 512,))(inputs=qa_context)


        # answer classification
        # answer_fc_1 = Dense(units=1000,
        #                     activation='relu',
        #                     kernel_regularizer=l2(0.001),
        #                     name='answer_fc_1')(inputs=qa_context)
        # # answer_fc_1 = Dropout(rate=0.5, seed=seed)(inputs=answer_fc_1)
        # answer_fc_2 = Dense(units=1000,
        #                     activation='relu',
        #                     kernel_regularizer=l2(0.001),
        #                     name='answer_fc_2')(inputs=answer_fc_1)
        # answer_fc_2 = Dropout(rate=0.5, seed=seed)(inputs=answer_fc_2)
        answer_pred = Dense(units=self.n_answers,
                            activation='softmax',
                            kernel_regularizer=l2(0),
                            name='answer_classifier')(inputs=qa_context)


        best_ans = Lambda(lambda x: K.argmax(x, axis=-1))(inputs=answer_pred)
        best_ans = Reshape(target_shape=(1,))(inputs=best_ans)
        best_ans = RepeatVector(n=10, )(inputs=best_ans)
        best_ans = Reshape(target_shape=(10,),
                           name='best_ans')(inputs=best_ans)


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

        print(self.model.metrics_names)