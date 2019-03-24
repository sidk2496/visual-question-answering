import h5py as h5
import pickle
import json
import numpy as np
from models import *
from datagen import *
from sklearn.model_selection import train_test_split
from numpy.random import seed
from tensorflow import set_random_seed


def main():

    seed(24)
    set_random_seed(24)

    batch_size = 32
    epochs = 10
    question_embed_dim = 256
    lstm_dim = 512
    n_answers = 1000
    n_train = 10

    dir_path = "../../data/"
    qa_data = h5.File(dir_path + "v2_data_prepro.h5", "r")
    img_feat = h5.File(dir_path + "v2_data_img.h5", "r")
    with open(dir_path + 'v2_data_prepro.json', 'r') as prepro_file:
        prepro_data = json.load(prepro_file)

    VOCAB_SIZE = len(prepro_data['ix_to_word'])
    MAX_QUESTION_LEN = qa_data['ques_train'].shape[1]

    questions_train, questions_val, ques_to_img_train, ques_to_img_val, answers_train, answers_val = \
        train_test_split(np.array(qa_data['ques_train'])[:n_train],
                         np.array(qa_data['img_pos_train'])[:n_train],
                         np.array(qa_data['answers'])[:n_train],
                         test_size=0.2,
                         random_state=24)

    train_datagen = DataGenerator(img_feat=np.array(img_feat['images_train']),
                                  questions=questions_train,
                                  answers=answers_train,
                                  ques_to_img=ques_to_img_train,
                                  VOCAB_SIZE=VOCAB_SIZE,
                                  n_answers=n_answers,
                                  batch_size=batch_size,
                                  shuffle=True)

    val_datagen = DataGenerator(img_feat=np.array(img_feat['images_train']),
                                questions=questions_val,
                                answers=answers_val,
                                ques_to_img=ques_to_img_val,
                                VOCAB_SIZE=VOCAB_SIZE,
                                n_answers=n_answers,
                                batch_size=batch_size,
                                shuffle=True)

    model = TimeDistributedCNN(question_embed_dim=question_embed_dim,
                               lstm_dim=lstm_dim,
                               n_answers=n_answers,
                               model_name='time_dist_cnn.h5',
                               VOCAB_SIZE=VOCAB_SIZE,
                               MAX_QUESTION_LEN=MAX_QUESTION_LEN)


    print("Starting training...")
    history = model.train(train_data=train_datagen,
                          val_data=val_datagen,
                          batch_size=batch_size,
                          epochs=epochs)
    print("Finished training.")

    print(history.history)

    with open('history.pkl', 'wb') as history_file:
        pickle.dump(history.history, history_file)

    with open('history.pkl', 'rb') as history_file:
        h = pickle.load(history_file)

    print(h)


if __name__ == '__main__':
    main()