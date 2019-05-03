import json
import argparse
import h5py as h5
import os
import numpy as np
import sys
import pickle
from models.img_ques_attention import ImgQuesAttentionNet
from models.conv_attention import ConvAttentionNet



def main(args):
    lstm_dim = 512
    n_answers = 1001
    question_embed_dim = 256
    idx = args.idx

    qa_data = h5.File(os.path.join(args.data_path, "data_prepro.h5"), "r")

    with open(os.path.join(args.data_path, "data_prepro.json"), "r") as file:
        prepro_data = json.load(file)

    img_feat = h5.File(os.path.join(args.data_path, "data_img.h5"), "r")['images_train']

    #####################################################################

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

    #####################################################################

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

    ix_to_ans = prepro_data['ix_to_ans']
    question_ids = np.array(qa_data['question_id_train']).tolist()

    # Define appropriate model
    if args.model_type == 'img_ques_att':
        model = ImgQuesAttentionNet(lstm_dim=lstm_dim,
                                    n_answers=n_answers,
                                    model_path=os.path.basename(args.model_path),
                                    VOCAB_SIZE=VOCAB_SIZE,
                                    MAX_QUESTION_LEN=MAX_QUESTION_LEN,
                                    question_embed_dim=question_embed_dim,
                                    log_path=None)

    elif args.model_type == 'conv_attention':
        model = ConvAttentionNet(lstm_dim=lstm_dim,
                                 n_answers=n_answers,
                                 model_path=os.path.basename(args.model_path),
                                 VOCAB_SIZE=VOCAB_SIZE,
                                 MAX_QUESTION_LEN=MAX_QUESTION_LEN,
                                 question_embed_dim=question_embed_dim,
                                 log_path=None)

    model.load_weights(weights_filename=args.model_path)

    test_input = [[img_feat[ques_to_img[idx] - 1]], [questions[idx, :-1]]]
    _, y_pred, _ = model.model.predict(x=test_input, batch_size=1)
    y_pred = np.argmax(y_pred[0])
    ix_to_word = prepro_data['ix_to_word']
    question = [ix_to_word[str(int(i))] for i in questions[idx] if i < len(ix_to_word) and i > 0]
    print("Question: ", question)
    print(np.sum(questions[idx][1:] != 0))
    print("Predicted answer:", ix_to_ans[str(int(y_pred + 1))])
    print("Actual answer:", ix_to_ans[str(int(answers[idx][0]))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['img_ques_att', 'show_n_tell',
                                                           'time_dist_cnn', 'ques_att',
                                                           'conv_attention'], help='type of model')
    parser.add_argument('--model_path', type=str, default='../models/model', help='path to model file')
    parser.add_argument('--data_path', type=str, default='../data/', help='path to input data')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size to use for testing')
    parser.add_argument('--idx', type=int, default=0)
    main(parser.parse_args())