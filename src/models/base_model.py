seed = 10707
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)

from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.applications.vgg19 import VGG19
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
    def __init__(self, lstm_dim, n_answers, question_embed_dim, VOCAB_SIZE, MAX_QUESTION_LEN, log_path, model_path):
        self.lstm_dim = lstm_dim
        self.n_answers = n_answers
        self.question_embed_dim = question_embed_dim
        self.VOCAB_SIZE = VOCAB_SIZE
        self.MAX_QUESTION_LEN = MAX_QUESTION_LEN
        self.log_path = log_path
        self.model_path = model_path
        K.set_image_data_format('channels_first')
        self.cnn = VGG19(include_top=True,
                         weights='imagenet',
                         input_tensor=None,
                         input_shape=None,
                         pooling=None,
                         classes=1000)
        # print(self.cnn.summary())


    def load_weights(self, weights_filename):
        self.model.load_weights(weights_filename)

    def train(self, train_data, val_data, batch_size=32, epochs=10):
        checkpoint = ModelCheckpoint(self.model_path + '.h5',
                                     monitor='val_best_ans_custom_acc',
                                     save_best_only=True,
                                     verbose=2)

        tensorboard = TensorBoard(log_dir=self.log_path,
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
        y_pred, _ = self.model.predict_generator(generator=test_data,
                                                 use_multiprocessing=False,
                                                 workers=cpu_count(),
                                                 verbose=0)
        return np.argmax(y_pred, axis=-1) + 1
