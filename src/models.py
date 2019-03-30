import numpy as np
np.random.seed(24)
import tensorflow as tf
tf.set_random_seed(24)
from multiprocessing import cpu_count


class VQA_Net:
    def __init__(self, question_embed_dim, lstm_dim, n_answers, model_name, VOCAB_SIZE, MAX_QUESTION_LEN):
        self.question_embed_dim = question_embed_dim
        self.lstm_dim = lstm_dim
        self.n_answers = n_answers
        self.model_name = model_name
        self.VOCAB_SIZE = VOCAB_SIZE
        self.MAX_QUESTION_LEN = MAX_QUESTION_LEN

    def load_weights(self, weights_filename):
        self.model.load_weights(weights_filename)

    def train(self, train_data, val_data, batch_size=32, epochs=10):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.model_name,
                                                        monitor='val_loss',
                                                        save_best_only=True,
                                                        verbose=2)

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../train_log',
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


class ShowNTell_Net(VQA_Net):
    def __init__(self, question_embed_dim, lstm_dim, n_answers, model_name, VOCAB_SIZE, MAX_QUESTION_LEN):
        super().__init__(question_embed_dim=question_embed_dim,
                         lstm_dim=lstm_dim,
                         n_answers=n_answers,
                         model_name=model_name,
                         VOCAB_SIZE=VOCAB_SIZE,
                         MAX_QUESTION_LEN=MAX_QUESTION_LEN)
        self.build()

    def build(self):
        # with tf.device('/cpu:0'):
        image_features = tf.keras.layers.Input(shape=(4096,),
                                               dtype='float32',
                                               name='image_input')

        image_embedding = tf.keras.layers.Dense(units=self.question_embed_dim,
                                                activation='elu',
                                                name='image_embedding')(inputs=image_features)

        reshape_image_embedding = tf.keras.layers.Reshape(target_shape=(1, self.question_embed_dim),
                                                  name='reshape_image_embedding')(image_embedding)

        question_input = tf.keras.layers.Input(shape=(self.MAX_QUESTION_LEN,),
                                               dtype='int32',
                                               name='question_input')

        question_input_masked = tf.keras.layers.Masking(mask_value=0,
                                                        input_shape=(self.MAX_QUESTION_LEN,),
                                                        name='question_input_masked')(inputs=question_input)

        question_embedding = tf.keras.layers.Embedding(input_dim=self.VOCAB_SIZE + 1,
                                                       output_dim=self.question_embed_dim,
                                                       input_length=self.MAX_QUESTION_LEN,
                                                       name='question_embedding')(inputs=question_input_masked)

        image_question_embedding = tf.keras.layers.Concatenate(axis=1,
                                                               name='image_question_embedding')(inputs=[reshape_image_embedding, question_embedding])

        question_features, last_h, _ = tf.keras.layers.LSTM(units=self.lstm_dim,
                                                            return_sequences=True,
                                                            return_state=True,
                                                            name='question_generator')(inputs=image_question_embedding)

        question_pred = tf.keras.layers.TimeDistributed(layer=tf.keras.layers.Dense(units=self.VOCAB_SIZE + 1,
                                                                                    activation='softmax'),
                                                        name='question_word_classifier')(inputs=question_features)

        question_pred_trunc_begin = tf.keras.layers.Lambda(lambda x: x[:, 1:, :],
                                                           name='question_word_classifier_trunc_begin')(inputs=question_pred)

        answer_fc_1 = tf.keras.layers.Dense(units=1000,
                                           activation='elu',
                                           name='answer_fc_1')(inputs=last_h)

        answer_fc_2 = tf.keras.layers.Dense(units=1000,
                                           activation='elu',
                                           name='answer_fc_2')(inputs=answer_fc_1)

        answer_pred = tf.keras.layers.Dense(units=self.n_answers,
                                            activation='softmax',
                                            name='answer_classifier')(inputs=answer_fc_2)

        self.model = tf.keras.Model(inputs=[image_features, question_input],
                                    outputs=[question_pred_trunc_begin, answer_pred])

        # self.model = tf.keras.utils.multi_gpu_model(model=model, gpus=1)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])


class TimeDistributedCNN_Net(VQA_Net):
    def __init__(self, question_embed_dim, lstm_dim, n_answers, model_name, VOCAB_SIZE, MAX_QUESTION_LEN):
        super().__init__(question_embed_dim=question_embed_dim,
                         lstm_dim=lstm_dim,
                         n_answers=n_answers,
                         model_name=model_name,
                         VOCAB_SIZE=VOCAB_SIZE,
                         MAX_QUESTION_LEN=MAX_QUESTION_LEN)

        self.build()

    def build(self):
        # with tf.device('/cpu:0'):
        image_features = tf.keras.layers.Input(shape=(4096,),
                                               dtype='float32',
                                               name='image_input')

        image_embedding = tf.keras.layers.Dense(units=self.question_embed_dim,
                                                activation='elu',
                                                name='image_embedding')(inputs=image_features)

        repeated_image_embedding = tf.keras.layers.RepeatVector(n=self.MAX_QUESTION_LEN,
                                                                name='repeated_image_embedding')(inputs=image_embedding)

        question_input = tf.keras.layers.Input(shape=(self.MAX_QUESTION_LEN,),
                                               dtype='int32',
                                               name='question_input')

        question_input_masked = tf.keras.layers.Masking(mask_value=0,
                                                        input_shape=(self.MAX_QUESTION_LEN,),
                                                        name='question_input_masked')(inputs=question_input)

        question_embedding = tf.keras.layers.Embedding(input_dim=self.VOCAB_SIZE + 1,
                                                       output_dim=self.question_embed_dim,
                                                       input_length=self.MAX_QUESTION_LEN,
                                                       name='question_embedding')(inputs=question_input_masked)

        image_question_embedding = tf.keras.layers.Concatenate(axis=-1,
                                                               name='image_question_embedding')(inputs=[repeated_image_embedding, question_embedding])

        question_features, last_h, _ = tf.keras.layers.LSTM(units=self.lstm_dim,
                                                            return_sequences=True,
                                                            return_state=True,
                                                            name='question_generator')(inputs=image_question_embedding)

        question_pred = tf.keras.layers.TimeDistributed(layer=tf.keras.layers.Dense(units=self.VOCAB_SIZE + 1,
                                                                                    activation='softmax'),
                                                        name='question_word_classifier')(inputs=question_features)

        answer_fc_1 = tf.keras.layers.Dense(units=1000,
                                            activation='elu',
                                            name='answer_fc_1')(inputs=last_h)

        answer_fc_2 = tf.keras.layers.Dense(units=1000,
                                            activation='elu',
                                            name='answer_fc_2')(inputs=answer_fc_1)

        answer_pred = tf.keras.layers.Dense(units=self.n_answers,
                                            activation='softmax',
                                            name='answer_classifier')(inputs=answer_fc_2)

        self.model = tf.keras.Model(inputs=[image_features, question_input],
                               outputs=[question_pred, answer_pred])

        # self.model = tf.keras.utils.multi_gpu_model(model=model, gpus=1)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])