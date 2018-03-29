# import pandas as pd
import numpy as np


def loadGloveModel(gloveFile):
    print("Loading Glove Model...")
    f = open(gloveFile, 'r')
    embedding_index = {}
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embedding_index))
    return embedding_index


def get_embedding_matrix(vocab, EMBEDDING_DIM, GloveDimOption=100, random_init=False):
    nb_words = len(vocab)
    # this  could be 50 (171.4 MB), 100 (347.1 MB), 200 (693.4 MB), or 300 (1 GB)
    # GloveDimOption = '100'
    assert GloveDimOption in set(
        [50, 100, 200, 300]), 'GloveDimOption should be 50, 100, 200 or 300'
    if GloveDimOption != 300:
        embeddings_index = loadGloveModel(
            './data/glove.6B.' + str(GloveDimOption) + 'd.txt')
    else:
        embeddings_index = loadGloveModel(
            './data/glove.840B.300d.txt')
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    null_embedding = 0
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        elif i != 0 and random_init:
            embedding_matrix[i] = np.random.uniform(-0.05, 0.05, EMBEDDING_DIM)
            null_embedding += 1
    # print('Null word embeddings: %d' %
    #       np.sum(np.sum(embedding_matrix, axis=1) == 0))
    print('Null word embeddings: %d' % null_embedding)
    return embedding_matrix
