import numpy as np
import tensorflow as tf
from tensorflow import set_random_seed
from numpy.random import seed

seed(24)
set_random_seed(24)

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_feat, questions, answers, ques_to_img, VOCAB_SIZE, n_answers, batch_size=32, shuffle=True):
        self.img_feat = img_feat
        self.questions = questions
        self.answers = answers
        self.ques_to_img = ques_to_img
        self.VOCAB_SIZE = VOCAB_SIZE
        self.n_answers = n_answers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_data = questions.shape[0]
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.n_data / self.batch_size))

    def __getitem__(self, idx):
        begin = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.n_data)
        X, y = self.__data_generation(self.indices[begin: end])
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(self.n_data)
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        X_ques = self.questions[indices]
        ques_to_img = self.ques_to_img[indices] - 1
        X_img = self.img_feat[ques_to_img]
        y_ques = tf.keras.utils.to_categorical(y=X_ques,
                                               num_classes=self.VOCAB_SIZE)
        y_ans = tf.keras.utils.to_categorical(y=self.answers[indices] - 1,
                                              num_classes=self.n_answers)
        return [X_img, X_ques], [y_ques, y_ans]