seed = 10707    

import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)
from keras.applications.vgg19 import preprocess_input
from keras.utils import Sequence

class DataGenerator(Sequence):
    # Initialize generator
    def __init__(self, img_feat, questions, answers, ques_to_img, VOCAB_SIZE, n_answers,
                 batch_size=32, split='train', extracted=True, shuffle=True):
        self.img_feat = img_feat
        self.questions = questions
        self.answers = answers
        self.ques_to_img = ques_to_img
        self.VOCAB_SIZE = VOCAB_SIZE
        self.n_answers = n_answers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_data = questions.shape[0]
        self.split = split
        self.extracted = extracted
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.n_data / self.batch_size))

    # Generate batch
    def __getitem__(self, idx):
        begin = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.n_data)
        X, y = self.__data_generation(self.indices[begin: end])
        return X, y

    # Shuffule after each epoch
    def on_epoch_end(self):
        self.indices = np.arange(self.n_data)
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    # Create batch
    def __data_generation(self, indices):
        X_ques = self.questions[indices, :-1]
        ques_to_img = self.ques_to_img[indices] - 1
        X_img = self.img_feat[ques_to_img]
        if not self.extracted:
            # CAUTION: Check position of channels
            X_img = preprocess_input(X_img, data_format='channels_first')
        if self.split in ['train', 'val']:
            y_ques = tf.keras.utils.to_categorical(y=self.questions[indices, 1:],
                                                   num_classes=self.VOCAB_SIZE)
            y_ans_best = tf.keras.utils.to_categorical(y=self.answers[indices, 0] - 1,
                                                       num_classes=self.n_answers)
            y_ans_top_10 = self.answers[indices, 1:] - 1
            return [X_img, X_ques], [y_ques, y_ans_best, y_ans_top_10]
        elif self.split == 'test':
            return [X_img, X_ques], []
        else:
            raise AssertionError
