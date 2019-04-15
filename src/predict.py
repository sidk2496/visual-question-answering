import json
import argparse
import h5py as h5
import os
import numpy as np

from models.img_ques_attention import ImgQuesAttentionNet
from models.show_n_tell import ShowNTellNet
from models.ques_attention import QuesAttentionShowNTellNet
from datagen import DataGenerator


def write_predictions(filepath, y_pred, ix_to_ans, question_ids):
    answers = [ix_to_ans[str(ix)] for ix in y_pred]
    qa_pairs = []
    for ques, ans in zip(question_ids, answers):
        qa_pairs.append({'answer': ans, 'question_id': ques})

    with open(filepath + '.json', 'w') as pred_file:
        json.dump(qa_pairs, pred_file)


def main(args):
    lstm_dim = 512
    n_answers = 1001
    question_embed_dim = 256

    qa_data = h5.File(os.path.join(args.data_path, "data_prepro.h5"), "r")
    img_feat = h5.File(os.path.join(args.data_path, "data_img.h5"), "r")
    with open(os.path.join(args.data_path, "data_prepro.json"), "r") as file:
        prepro_data = json.load(file)

    VOCAB_SIZE = len(prepro_data['ix_to_word'])
    MAX_QUESTION_LEN = qa_data['ques_test'].shape[1]
    SOS = VOCAB_SIZE + 1
    # Add 1 for SOS and 1 for '0' -> padding
    VOCAB_SIZE += 2

    # Add SOS char at the beginning for every question
    questions = np.zeros((qa_data['ques_test'].shape[0], MAX_QUESTION_LEN + 1))
    questions[:, 1:] = qa_data['ques_test']
    questions[:, 0] = SOS

    ques_to_img = np.array(qa_data['img_pos_test'])

    ix_to_ans = prepro_data['ix_to_ans']
    question_ids = np.array(qa_data['question_id_test']).tolist()
    n_test = len(question_ids)

    # Load appropriate model
    if args.model_type == 'img_ques_attention':
        model = ImgQuesAttentionNet(lstm_dim=lstm_dim,
                                    n_answers=n_answers,
                                    model_name=os.path.basename(args.model_path),
                                    VOCAB_SIZE=VOCAB_SIZE,
                                    MAX_QUESTION_LEN=MAX_QUESTION_LEN,
                                    question_embed_dim=question_embed_dim)
    elif args.model_type == 'show_n_tell':
        model = ShowNTellNet(lstm_dim=lstm_dim,
                             n_answers=n_answers,
                             model_name=os.path.basename(args.model_path),
                             VOCAB_SIZE=VOCAB_SIZE,
                             MAX_QUESTION_LEN=MAX_QUESTION_LEN,
                             question_embed_dim=question_embed_dim)
    elif args.model_type == 'ques_attention':
        model = QuesAttentionShowNTellNet(lstm_dim=lstm_dim,
                                          n_answers=n_answers,
                                          model_name=os.path.basename(args.model_path),
                                          VOCAB_SIZE=VOCAB_SIZE,
                                          MAX_QUESTION_LEN=MAX_QUESTION_LEN,
                                          question_embed_dim=question_embed_dim)

    model.load_weights(weights_filename=args.model_path)

    chunk_size = 1000000000
    y_pred = np.zeros(n_test, dtype=np.int)
    n_chunks = len(range(0, n_test, chunk_size))
    for i, batch in enumerate(range(0, n_test, chunk_size)):
        begin = batch
        end = min(n_test, batch + chunk_size)
        # Test data generator
        test_datagen = DataGenerator(img_feat=np.array(img_feat['images_test']),
                                     questions=questions[begin: end],
                                     answers=[],
                                     ques_to_img=ques_to_img[begin: end],
                                     VOCAB_SIZE=VOCAB_SIZE,
                                     n_answers=n_answers,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     split='test')
        y_pred_chunk = model.predict(test_data=test_datagen)
        if i + 1 % 50 == 0:
            print("Completed testing on {}/{} chunks...".format(i + 1, n_chunks))
        y_pred[begin: end] = y_pred_chunk

    write_predictions(filepath=args.dest_path,
                      y_pred=y_pred,
                      ix_to_ans=ix_to_ans,
                      question_ids=question_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['img_ques_attention', 'show_n_tell',
                                                           'ques_attention'], help='type of model')
    parser.add_argument('--model_path', type=str, default='../models/model.h5', help='path to model file')
    parser.add_argument('--data_path', type=str, default='../data/', help='path to input data')
    parser.add_argument('--dest_path', type=str, help='prediciton file full path (without the file extension)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size to use for testing')
    main(parser.parse_args())