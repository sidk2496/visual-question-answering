seed = 10707

import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)

import os
import h5py as h5
import pickle
import json
import sys

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from models.show_n_tell import ShowNTellNet
from models.ques_attention import QuesAttentionShowNTellNet
from models.img_ques_attention import ImgQuesAttentionNet
from models.conv_attention import ConvAttentionNet
from models.time_dist_cnn import TimeDistributedCNNNet
from datagen import *

import argparse


def main(args):
    # from client import device_lib

    # print(device_lib.list_local_devices())
    # Set parameters for network
    question_embed_dim = 256
    lstm_dim = 512
    n_answers = 1001


    n_train = 500


    # Read QA data
    qa_data = h5.File(os.path.join(args.data_path, "data_prepro.h5"), "r")
    with open(os.path.join(args.data_path, "data_prepro.json"), "r") as prepro_file:
        prepro_data = json.load(prepro_file)

    # Read V data
    if args.extracted:
        img_feat = np.array(h5.File(os.path.join(args.data_path, "data_img.h5"), "r")['images_train'])
    else:
        print("Loading images...\n")
        img_feat = [img_to_array(load_img(os.path.join(args.data_path, image_filename), target_size=(224, 224)), dtype='uint8', data_format='channels_first')
                    for image_filename in prepro_data['unique_img_train']]
        img_feat = np.array(img_feat, dtype=np.uint8)

    # Some preprocessing
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

    # Load answers
    print("\nStarting to load answers...")
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

        with open(os.path.join(args.data_path, 'v2_mscoco_train2014_annotations.json'), 'r') as annot_file:
            train_annotations = json.load(annot_file)['annotations']
        with open(os.path.join(args.data_path, 'v2_mscoco_val2014_annotations.json'), 'r') as annot_file:
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
    print("Finished loading answers!\n")

    # Train-val random split
    print("\nCreating train-val split...")
    questions_train, questions_val, ques_to_img_train, ques_to_img_val, answers_train, answers_val = \
        train_test_split(questions,
                         ques_to_img,
                         answers,
                         test_size=0.2,
                         random_state=seed)
    print("Created train-val split!\n")

    # Create generators for training and validation
    print("\nCreating generators...")
    # Train data generator
    train_datagen = DataGenerator(img_feat=img_feat,
                                  questions=questions_train,
                                  answers=answers_train,
                                  ques_to_img=ques_to_img_train,
                                  VOCAB_SIZE=VOCAB_SIZE,
                                  n_answers=n_answers,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  split='train',
                                  extracted=args.extracted)

    # Validation data generator
    val_datagen = DataGenerator(img_feat=img_feat,
                                questions=questions_val,
                                answers=answers_val,
                                ques_to_img=ques_to_img_val,
                                VOCAB_SIZE=VOCAB_SIZE,
                                n_answers=n_answers,
                                batch_size=args.batch_size,
                                shuffle=True,
                                split='val',
                                extracted=args.extracted)

    print("Created generators!\n")

    # Define appropriate model
    print("\nDefining  model...")
    if args.model_type == 'img_ques_attention':
        model = ImgQuesAttentionNet(lstm_dim=lstm_dim,
                                    n_answers=n_answers,
                                    VOCAB_SIZE=VOCAB_SIZE,
                                    MAX_QUESTION_LEN=MAX_QUESTION_LEN,
                                    question_embed_dim=question_embed_dim,
                                    log_path=args.log_path,
                                    model_path=args.model_path)
    elif args.model_type == 'show_n_tell':
        model = ShowNTellNet(lstm_dim=lstm_dim,
                             n_answers=n_answers,
                             VOCAB_SIZE=VOCAB_SIZE,
                             MAX_QUESTION_LEN=MAX_QUESTION_LEN,
                             question_embed_dim=question_embed_dim,
                             log_path=args.log_path,
                             model_path=args.model_path)
    elif args.model_type == 'ques_attention':
        model = QuesAttentionShowNTellNet(lstm_dim=lstm_dim,
                                          n_answers=n_answers,
                                          VOCAB_SIZE=VOCAB_SIZE,
                                          MAX_QUESTION_LEN=MAX_QUESTION_LEN,
                                          question_embed_dim=question_embed_dim,
                                          log_path=args.log_path,
                                          model_path=args.model_path)
    elif args.model_type == 'conv_attention':
        model = ConvAttentionNet(lstm_dim=lstm_dim,
                                 n_answers=n_answers,
                                 VOCAB_SIZE=VOCAB_SIZE,
                                 MAX_QUESTION_LEN=MAX_QUESTION_LEN,
                                 question_embed_dim=question_embed_dim,
                                 log_path=args.log_path,
                                 model_path=args.model_path)
    elif args.model_type == 'time_dist_cnn':
        model = TimeDistributedCNNNet(lstm_dim=lstm_dim,
                                      n_answers=n_answers,
                                      VOCAB_SIZE=VOCAB_SIZE,
                                      MAX_QUESTION_LEN=MAX_QUESTION_LEN,
                                      question_embed_dim=question_embed_dim,
                                      log_path=args.log_path,
                                      model_path=args.model_path)


    print(model.model.summary())
    print("Model ready!\n")

    if args.load_model != None:
        model.load_weights(args.load_model)
        print("Loaded model weights\n")

    # Train model
    print("\nStarting training...")
    model.train(train_data=train_datagen,
                          val_data=val_datagen,
                          batch_size=args.batch_size,
                          epochs=args.epochs)
    print("Finished training!\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--data_path', type=str, default='../data/', help='directory for training data')
    parser.add_argument('--model_type', type=str, choices=['img_ques_attention', 'show_n_tell',
                                                           'ques_attention', 'conv_attention',
                                                           'time_dist_cnn'], help='type of model to train')
    parser.add_argument('--log_path', type=str, default='../train_log/', help='tensorboard logdir')
    parser.add_argument('--model_path', type=str, default='../models/model', help='path to save model without file extension')
    parser.add_argument('--load_model', type=str, default=None, help='path to load model without file extension')
    parser.add_argument('--extracted', action='store_true', help='True for reading extracted features False for reading raw images')
    main(parser.parse_args())