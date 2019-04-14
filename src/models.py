seed = 10707
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)

from tensorflow.python.keras.layers import *
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras import Model
from tensorflow.python.keras.activations import softmax
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import optimizers

# from attention import Attention

from multiprocessing import cpu_count



def dummy(y_true, y_pred):
    return K.zeros(1)


def custom_acc(y_true, y_pred):
    best_pred_count = K.sum(K.cast(K.equal(y_true, y_pred),
                                   dtype='float32'),
                            axis=-1)
    acc = K.minimum(best_pred_count / 3, 1.0)
    return K.mean(acc)

class VQANet:
    def __init__(self, lstm_dim, n_answers, model_name, VOCAB_SIZE, MAX_QUESTION_LEN, question_embed_dim=None):
        self.question_embed_dim = question_embed_dim
        self.lstm_dim = lstm_dim
        self.n_answers = n_answers
        self.model_name = model_name
        self.VOCAB_SIZE = VOCAB_SIZE
        self.MAX_QUESTION_LEN = MAX_QUESTION_LEN

    def load_weights(self, weights_filename):
        self.model.load_weights(weights_filename)

    def train(self, train_data, val_data, batch_size=32, epochs=10):
        checkpoint = ModelCheckpoint(self.model_name,
                                     monitor='val_best_ans_custom_acc',
                                     save_best_only=True,
                                     verbose=2)

        tensorboard = TensorBoard(log_dir='../train_log',
                                  write_graph=True,
                                  batch_size=batch_size)

        callbacks = [checkpoint, tensorboard]

        history = self.model.fit_generator(generator=train_data,
                                           validation_data=val_data,
                                           epochs=epochs,
                                           verbose=1,
                                           callbacks=callbacks,
                                           workers=cpu_count(),
                                           use_multiprocessing=False)
        return history

    def predict(self, test_data):
        _, y_pred, _ = self.model.predict_generator(generator=test_data,
                                                    use_multiprocessing=False,
                                                    workers=cpu_count(),
                                                    verbose=1)
        return np.argmax(y_pred, axis=-1) + 1



class ShowNTellNet(VQANet):
    def __init__(self, lstm_dim, n_answers, model_name, VOCAB_SIZE, MAX_QUESTION_LEN, question_embed_dim=None):
        super().__init__(question_embed_dim=question_embed_dim,
                         lstm_dim=lstm_dim,
                         n_answers=n_answers,
                         model_name=model_name,
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
        image_embedding = Reshape(target_shape=(1, self.question_embed_dim))(inputs=image_embedding)


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
                                               name='image_question_embedding')(inputs=[image_embedding,
                                                                                        question_embedding])
        question_embedding, last_h, _ = LSTM(units=self.lstm_dim,
                                             return_sequences=True,
                                             return_state=True,
                                             name='question_lstm')(inputs=image_question_embedding)
        question_pred = TimeDistributed(layer=Dense(units=self.VOCAB_SIZE,
                                                    activation='softmax'))(inputs=question_embedding)
        question_pred = Lambda(lambda x: x[:, 1:, :],
                               name='question_classifier')(inputs=question_pred)


        answer_fc_1 = Dense(units=1000,
                            activation='relu',
                            name='answer_fc_1')(inputs=last_h)
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




class TimeDistributedCNNNet(VQANet):
    def __init__(self, lstm_dim, n_answers, model_name, VOCAB_SIZE, MAX_QUESTION_LEN, question_embed_dim=None):
        super().__init__(question_embed_dim=question_embed_dim,
                         lstm_dim=lstm_dim,
                         n_answers=n_answers,
                         model_name=model_name,
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




class QuesAttentionShowNTellNet(VQANet):
    def __init__(self, lstm_dim, n_answers, model_name, VOCAB_SIZE, MAX_QUESTION_LEN, question_embed_dim=None):
        super().__init__(question_embed_dim=question_embed_dim,
                         lstm_dim=lstm_dim,
                         n_answers=n_answers,
                         model_name=model_name,
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
        image_embedding = Reshape(target_shape=(1, self.question_embed_dim))(inputs=image_embedding)


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
                                               name='image_question_embedding')(inputs=[image_embedding,
                                                                                        question_embedding])
        question_embedding = LSTM(units=self.lstm_dim,
                                  return_sequences=True,
                                  name='question_lstm')(inputs=image_question_embedding)
        # question_pred = TimeDistributed(layer=Dense(units=self.VOCAB_SIZE, activation='softmax'),
        #                                 name='question_classifier')(inputs=question_embedding)


        attention_weights = TimeDistributed(layer=Dense(units=1),
                                            name='attention')(inputs=question_embedding)
        attention_weights = Lambda(lambda x: softmax(x, axis=1),
                                   name='attention_softmax')(inputs=attention_weights)
        attention_mul = Multiply()(inputs=[attention_weights, question_embedding])
        context = Lambda(lambda x: K.sum(x, axis=1), name='context')(inputs=attention_mul)


        answer_fc_1 = Dense(units=1000,
                            activation='relu',
                            name='answer_fc_1')(inputs=context)
        answer_fc_2 = Dense(units=1000,
                            activation='relu',
                            name='answer_fc_2')(inputs=answer_fc_1)
        answer_pred = Dense(units=self.n_answers,
                            activation='softmax',
                            name='answer_classifier')(inputs=answer_fc_2)


        best_ans = Lambda(lambda x: K.argmax(x, axis=-1))(inputs=answer_pred)
        best_ans = Reshape(target_shape=(1,))(inputs=best_ans)
        best_ans = RepeatVector(n=10,)(inputs=best_ans)
        best_ans = Reshape(target_shape=(10,))(inputs=best_ans)


        self.model = Model(inputs=[image_input, question_input],
                           outputs=[answer_pred, best_ans])
        losses = {
            # 'ques_word_classifier': 'categorical_crossentropy',
            'answer_classifier': 'categorical_crossentropy',
            'best_ans_repeat10': dummy
        }
        metrics = {
            # 'ques_word_classifier': 'acc',
            'answer_classifier': 'acc',
            'best_ans_repeat10': custom_acc,
        }
        self.model.compile(loss=losses,
                           optimizer='adam',
                           metrics=metrics)




class ImgQuesAttentionNet(VQANet):
    def __init__(self, lstm_dim, n_answers, model_name, VOCAB_SIZE, MAX_QUESTION_LEN, question_embed_dim=None):
        super().__init__(question_embed_dim=question_embed_dim,
                         lstm_dim=lstm_dim,
                         n_answers=n_answers,
                         model_name=model_name,
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
                                                      return_sequences=True,
                                                      kernel_regularizer=l2(0.001)),
                                           name='question_lstm_1')(inputs=question_embedding)
        question_embedding = Bidirectional(layer=LSTM(units=self.lstm_dim,
                                                      return_sequences=True,
                                                      kernel_regularizer=l2(0.001)),
                                           name='question_lstm_2')(inputs=question_embedding)


        vq_attention_weights = TimeDistributed(layer=Dense(units=2 * self.lstm_dim,
                                                           kernel_regularizer=l2(0.001)))(inputs=image_feat)
        vq_attention_weights = Lambda(lambda x: K.batch_dot(*x, axes=[2, 2]))(inputs=[question_embedding,
                                                                                      vq_attention_weights])
        vq_attention_weights = Lambda(lambda x: softmax(x, axis=-1),
                                      name='vq_attention_weights')(inputs=vq_attention_weights)
        vq_context = Lambda(lambda x: K.batch_dot(*x, axes=[2, 1]),
                            name='vq_context')(inputs=[vq_attention_weights, image_feat])
        vq_context_question_embedding = Concatenate(axis=-1)(inputs=[vq_context, question_embedding])


        question_pred = TimeDistributed(layer=Dense(units=self.VOCAB_SIZE,
                                                    activation='softmax',
                                                    kernel_regularizer=l2(0.001)),
                                        name='question_classifier')(inputs=vq_context_question_embedding)


        qa_attention_weights = TimeDistributed(layer=Dense(units=1000,
                                                           activation='relu',
                                                           kernel_regularizer=l2(0.001)))(inputs=vq_context_question_embedding)
        # qa_attention_weights = Dropout(rate=0.5, seed=seed)(inputs=qa_attention_weights)
        qa_attention_weights = TimeDistributed(layer=Dense(units=256,
                                                           activation='relu',
                                                           kernel_regularizer=l2(0.001)))(inputs=qa_attention_weights)
        # qa_attention_weights = Dropout(rate=0.5, seed=seed)(inputs=qa_attention_weights)
        qa_attention_weights = TimeDistributed(layer=Dense(units=1,
                                                           kernel_regularizer=l2(0.001)))(inputs=qa_attention_weights)
        qa_attention_weights = Lambda(lambda x: softmax(x, axis=1),
                                      name='qa_attention_weights')(inputs=qa_attention_weights)
        qa_context = Dot(axes=1)(inputs=[qa_attention_weights, vq_context_question_embedding])
        qa_context = Reshape(target_shape=(2 * self.lstm_dim + 512,))(inputs=qa_context)


        answer_fc_1 = Dense(units=1000,
                            activation='relu',
                            kernel_regularizer=l2(0.001),
                            name='answer_fc_1')(inputs=qa_context)
        # answer_fc_1 = Dropout(rate=0.5, seed=seed)(inputs=answer_fc_1)
        answer_fc_2 = Dense(units=1000,
                            activation='relu',
                            kernel_regularizer=l2(0.001),
                            name='answer_fc_2')(inputs=answer_fc_1)
        # answer_fc_2 = Dropout(rate=0.5, seed=seed)(inputs=answer_fc_2)
        answer_pred = Dense(units=self.n_answers,
                            activation='softmax',
                            kernel_regularizer=l2(0.001),
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