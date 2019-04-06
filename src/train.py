seed = 10707

import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)

import h5py as h5
import pickle
import json
import sys

from models import *
from datagen import *
from sklearn.model_selection import train_test_split


def main():
    # from tensorflow.python.client import device_lib

    # print(device_lib.list_local_devices())

    batch_size = 512
    epochs = 100
    question_embed_dim = 256
    lstm_dim = 512
    n_answers = 1001

    # Read VQA data
    dir_path = "../data/"
    qa_data = h5.File(dir_path + "data_prepro.h5", "r")
    img_feat = h5.File(dir_path + "data_img.h5", "r")
    with open(dir_path + "data_prepro.json", "r") as prepro_file:
        prepro_data = json.load(prepro_file)
    
    VOCAB_SIZE = len(prepro_data['ix_to_word'])
    MAX_QUESTION_LEN = qa_data['ques_train'].shape[1]
    SOS = VOCAB_SIZE + 1
    # Add 1 for SOS and 1 for '0' -> padding
    VOCAB_SIZE += 2

    # Add SOS char at the beginning for every question
    questions = np.zeros((qa_data['ques_train'].shape[0], MAX_QUESTION_LEN + 1))
    questions[:, 1:] = qa_data['ques_train']
    questions[:, 0] = SOS

    ques_to_img = np.array(qa_data['img_pos_train'])

    print("Starting to load answers...")
    try:
        with open('../data/answers.pkl', 'rb') as answers_file:
            answers = pickle.load(answers_file)
    except FileNotFoundError:
        # Load all 10 answers per question
        answers = np.zeros((len(qa_data['answers']), 11))

        # best answer at idx 0
        answers[:, 0] = qa_data['answers']

        ques_id_to_ix = {ques_id: ix for ix, ques_id in enumerate(qa_data['question_id_train'])}
        ans_to_ix = {ans: int(ix) for ix, ans in prepro_data['ix_to_ans'].items()}

        with open(dir_path + 'v2_mscoco_train2014_annotations.json', 'r') as annot_file:
            train_annotations = json.load(annot_file)['annotations']
        with open(dir_path + 'v2_mscoco_val2014_annotations.json', 'r') as annot_file:
            val_annotations = json.load(annot_file)['annotations']

        for annot_num, annotation in enumerate(train_annotations):
            ques_id = annotation['question_id']
            if ques_id in ques_id_to_ix.keys():
                ques_ix = ques_id_to_ix[ques_id]
                for answer_num, answer in enumerate(annotation['answers']):
                    answers[ques_ix, answer_num + 1] = ans_to_ix.get(answer['answer'], n_answers)
            if (annot_num + 1) % 10000 == 0:
                sys.stdout.write("Completed processing {0:6d} annotations...\r".format(annot_num + 1))
        sys.stdout.write("Completed processing {0:6d} annotations...\r".format(annot_num + 1))

        for annot_num, annotation in enumerate(val_annotations):
            ques_id = annotation['question_id']
            if ques_id in ques_id_to_ix.keys():
                ques_ix = ques_id_to_ix[ques_id]
                for answer_num, answer in enumerate(annotation['answers']):
                    answers[ques_ix, answer_num + 1] = ans_to_ix.get(answer['answer'], n_answers)
            if (annot_num + 1) % 10000 == 0:
                sys.stdout.write("Completed processing {0:6d} annotations...\r".format(annot_num + 1))
        sys.stdout.write("Completed processing {0:6d} annotations...\n".format(annot_num + 1))

        with open('../data/answers.pkl', 'wb') as answers_file:
            pickle.dump(answers, answers_file)
    print("Finished loading answers!")

    print("Creating train-val split...")
    # Train-val random split
    questions_train, questions_val, ques_to_img_train, ques_to_img_val, answers_train, answers_val = \
        train_test_split(questions,
                         ques_to_img,
                         answers,
                         test_size=0.2,
                         random_state=seed)
    print("Created train-val split!")

    print("Creating generators...")
    # Train data generator
    train_datagen = DataGenerator(img_feat=np.array(img_feat['images_train']),
                                  questions=questions_train,
                                  answers=answers_train,
                                  ques_to_img=ques_to_img_train,
                                  VOCAB_SIZE=VOCAB_SIZE,
                                  n_answers=n_answers,
                                  batch_size=batch_size,
                                  shuffle=True)

    # Validation data generator
    val_datagen = DataGenerator(img_feat=np.array(img_feat['images_train']),
                                questions=questions_val,
                                answers=answers_val,
                                ques_to_img=ques_to_img_val,
                                VOCAB_SIZE=VOCAB_SIZE,
                                n_answers=n_answers,
                                batch_size=batch_size,
                                shuffle=True)
    print("Created generators!")

    print("Defining  model...")
    # Define model
    model = AttentionShowNTell(question_embed_dim=question_embed_dim,
                          lstm_dim=lstm_dim,
                          n_answers=n_answers,
                          model_name='../models/show_n_tell.h5',
                          VOCAB_SIZE=VOCAB_SIZE,
                          MAX_QUESTION_LEN=MAX_QUESTION_LEN)

    print(model.model.summary())
    print("Model ready!")

    # Train model
    print("Starting training...")
    history = model.train(train_data=train_datagen,
                          val_data=val_datagen,
                          batch_size=batch_size,
                          epochs=epochs)
    print("Finished training!")

    # with open('history.pkl', 'wb') as history_file:
    #     pickle.dump(history.history, history_file)


if __name__ == '__main__':
    main()