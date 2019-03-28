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
    from tensorflow.python.client import device_lib

    # print(device_lib.list_local_devices())
    seed(24)
    set_random_seed(24)

    batch_size = 32
    epochs = 10
    question_embed_dim = 256
    lstm_dim = 512
    n_answers = 1000
    n_train = 40000

    dir_path = "../../data/"
    qa_data = h5.File(dir_path + "data_prepro.h5", "r")
    img_feat = h5.File(dir_path + "data_img.h5", "r")
    with open(dir_path + 'data_prepro.json', 'r') as prepro_file:
        prepro_data = json.load(prepro_file)
    
    VOCAB_SIZE = len(prepro_data['ix_to_word'])
    MAX_QUESTION_LEN = qa_data['ques_train'].shape[1]
    SOS = VOCAB_SIZE + 1
    VOCAB_SIZE += 1

    questions = np.zeros((qa_data['ques_train'].shape[0], MAX_QUESTION_LEN + 1))
    questions[:, 1:] = qa_data['ques_train']
    questions[:, 0] = SOS

    ques_to_img = np.array(qa_data['img_pos_train'])
    answers = np.array(qa_data['answers'])

    questions_train, questions_val, ques_to_img_train, ques_to_img_val, answers_train, answers_val = \
        train_test_split(questions[:n_train],
                         ques_to_img[:n_train],
                         answers[:n_train],
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
                               model_name='../../models/time_dist_cnn.h5',
                               VOCAB_SIZE=VOCAB_SIZE,
                               MAX_QUESTION_LEN=MAX_QUESTION_LEN)

    print(model.model.summary())

    print("Starting training...")
    history = model.train(train_data=train_datagen,
                          val_data=val_datagen,
                          batch_size=batch_size,
                          epochs=epochs)
    print("Finished training.")

    with open('history.pkl', 'wb') as history_file:
        pickle.dump(history.history, history_file)


if __name__ == '__main__':
    main()