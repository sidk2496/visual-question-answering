seed = 10707    

import numpy as np
np.random.seed(seed)

import io
import pickle
import json

def gen_embedding_matrix(filename):
    # Read VQA data
    dir_path = "../data/"
    with open(dir_path + 'data_prepro.json', 'r') as prepro_file:
        prepro_data = json.load(prepro_file)

    # Define reverse mapping form word to ix
    word_to_ix = {word: int(ix) for ix, word in prepro_data['ix_to_word'].items()}
    # Add 1 for SOS and 1 for '0' -> padding
    VOCAB_SIZE = len(word_to_ix) + 2

    # Process embedding file
    fin = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    _, EMBEDDING_DIM = map(int, fin.readline().split())

    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    for line in fin:
        tokens = line.rstrip().split(' ')
        # print(tokens[0])
        if tokens[0] in word_to_ix.keys():
            embedding_matrix[word_to_ix[tokens[0]]] = np.array([float(val) for val in tokens[1:]])

    # Altering for SOS token
    embedding_matrix[-1] = 0.01 * np.ones(EMBEDDING_DIM)

    # Save embedding matrix
    # with open('../data/embed_fasttext_crawl.pkl', 'wb') as embed_file:
        pickle.dump(embedding_matrix, embed_file)

    return embedding_matrix