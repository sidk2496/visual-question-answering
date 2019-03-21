import h5py as h5
import numpy as np
from models import *


def main():
    batch_size = 32
    epochs = 100
    question_embed_dim = 256
    lstm_dim = 512
    n_answers = 1000
    n_train = 1000

    ques = h5.File("../../data/data_train_val/data_prepro.h5", "r")
    ques_train = ques['ques_train'][:n_train]
    ques_train_one_hot = tf.keras.utils.to_categorical(y=ques_train,
                                                       num_classes=VOCAB_SIZE)
    ques_to_image_train = ques['img_pos_train'][:n_train] - 1

    ans_train = ques['answers'][:n_train]
    ans_train_one_hot = tf.keras.utils.to_categorical(y=ans_train,
                                                      num_classes=n_answers)
    img_feat = h5.File("../../data/data_train_val/data_img.h5", "r")
    img_train = np.array(img_feat['images_train'])[ques_to_image_train]

    model = VQANet(question_embed_dim=question_embed_dim,
                         lstm_dim=lstm_dim,
                         n_answers=n_answers)

    print("Starting training...")
    model.train(x_train=[img_train, ques_train],
                y_train=[ques_train_one_hot, ans_train_one_hot],
                x_val=[],
                y_val=[],
                batch_size=batch_size,
                epochs=epochs)
    print("Finished training.")


if __name__ == '__main__':
    main()

