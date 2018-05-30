import os

import numpy as np
import pandas as pd
from keras import optimizers
from utils.data_utils import get_embedding_matrix
from model_zoo_for_bidaf import BIDAF_Model

from utils.train_utils import Trainer_helper
from utils.evaluate_utils import Evaluate_helper

import tensorflow as tf


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_NAME = 'bidaf_satt'

log_dir = './logs/'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
if not os.path.exists('./outputs/'):
    os.mkdir('./outputs/')
if not os.path.exists('./model/'):
    os.mkdir('./model/')
output_name = './outputs/squad_' + MODEL_NAME + '.pkl'

GloveDimOption = 100
EMBEDDING_DIM = 100

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1


def load_data(mode='train'):
    X = pd.read_pickle('./input/' + mode + '_context_vec.pkl')
    X_char = pd.read_pickle('./input/' + mode + '_char_vec.pkl')
    Xq = pd.read_pickle('./input/' + mode + '_question_vec.pkl')
    Xq_char = pd.read_pickle('./input/' + mode + '_question_char_vec.pkl')
    if mode == 'train':
        XYBegin = pd.read_pickle('./input/' + mode + '_ybegin_vec.pkl')
        XYEnd = pd.read_pickle('./input/' + mode + '_yend_vec.pkl')
        return X, X_char, Xq, Xq_char, XYBegin, XYEnd
    return X, X_char, Xq, Xq_char


def filter_longer(X, X_char, Xq, Xq_char, XYBegin, XYEnd, train_ids):
    dels = []
    for i, x in enumerate(X):
        if len(x) > 300:
            dels.append(i)
    for i, iid in enumerate(dels):
        del X[iid - i]
        del X_char[iid - i]
        del Xq[iid - i]
        del Xq_char[iid - i]
        del XYBegin[iid - i]
        del XYEnd[iid - i]
        del train_ids[iid - i]
    return X, X_char, Xq, Xq_char, XYBegin, XYEnd, train_ids


def train_valid_split(samples, VALIDATION_SPLIT=0.1):
    idx = np.arange(len(samples))
    np.random.seed(1024)
    np.random.shuffle(idx)
    idx_train = idx[:int(len(samples) * (1 - VALIDATION_SPLIT))]
    idx_val = idx[int(len(samples) * (1 - VALIDATION_SPLIT)):]
    return idx_train, idx_val


def combine_input_data(question_words, context_words, question_char, context_char, ids, mode='train'):
    ds = []
    ds.append(question_words)
    ds.append(question_char)
    ds.append(context_words)
    ds.append(context_char)
    if mode == 'train':
        ds.append(ids)
    return ds


def prepare_model(training=True, test_code=False, load_weights=False, lr=0.001, ft_model_path=None, epoch=None, need_filter=False):
    train_words, train_chars, train_q_words, train_q_chars, train_y_begin, train_y_end = load_data(
        'train')
    train_ids = pd.read_pickle('./input/train_question_id.pkl')
    if need_filter:
        train_words, train_chars, train_q_words, train_q_chars, train_y_begin, train_y_end, train_ids = filter_longer(
            train_words, train_chars, train_q_words, train_q_chars, train_y_begin, train_y_end, train_ids)

    dev_words, dev_chars, dev_q_words, dev_q_chars = load_data('dev')

    vocab2id = pd.read_pickle('./input/word_vocab.pkl')
    char2ids = pd.read_pickle('./input/char_vocab.pkl')
    embedding_matrix = get_embedding_matrix(
        vocab2id, EMBEDDING_DIM, GloveDimOption)

    train_data = combine_input_data(
        train_q_words, train_words, train_q_chars, train_chars, train_ids, 'train')
    train_y = [train_y_begin, train_y_end]
    dev_data = combine_input_data(
        dev_q_words, dev_words, dev_q_chars, dev_chars, None, 'dev')

    # train model
    bst_model_path = './model/squad_' + MODEL_NAME + '.hdf5'
    if epoch is None:
        weights_path = bst_model_path
    else:
        weights_path = './model/squad_' + MODEL_NAME + '.ep%d.hdf5' % epoch

    # with tf.device('/cpu:0'):
    model = BIDAF_Model(len(vocab2id), len(char2ids), embedding_matrix, use_highway=False,
                        user_char_embed=True, use_sefatt=True)


    model.summary()
    # rms = optimizers.Adadelta(lr=(lr / 10) if training and load_weights else lr)  # default
    rms = optimizers.Adam(lr=(lr / 10) if training and load_weights else lr)  # default
    model.compile(optimizer=rms, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    evaluater = Evaluate_helper()
    trainer = Trainer_helper(model, log_dir='./logs/' + MODEL_NAME,
                             bst_model_path=bst_model_path, output_name=output_name, evaluate_helper=evaluater)

    if load_weights:
        if training and ft_model_path is not None:
            trainer.load_weights(ft_model_path)
        else:
            trainer.load_weights(weights_path)

    if test_code:
        train_data = [d[:BATCH_SIZE * 2] for d in train_data]
        train_y = [d[:BATCH_SIZE * 2] for d in train_y]

    if training:
        trainer.fit(train_data, train_y, dev_data, batch_size=BATCH_SIZE, dev_batch_size=BATCH_SIZE//3, n_epoch=50, early_stop=5,
                    verbose_train=100, is_save_intermediate=False, method=['group_by_length'])
    return trainer, dev_data


def predict(text=False):
    trainer, dev_data = prepare_model(training=False, test_code=False, load_weights=True, lr=0.5)
    if not text:
        return trainer.predict_probs(dev_data, batch_size=70)
    return trainer.predict(dev_data, batch_size=70)


def finetune(ft_model_path=None):
    prepare_model(training=True, test_code=False, load_weights=True, lr=0.001, ft_model_path=ft_model_path)


def evaluate(epoch=None):
    trainer, dev_data = prepare_model(training=False, test_code=False, load_weights=True, lr=0.5)
    em, f1 = trainer.evaluate_on_dev(dev_data, BATCH_SIZE//3, -1)  # can control huge data
    print('--Dev Extract Match score :%f--------F1 score:%f' % (em, f1))


if __name__ == '__main__':
    # training
    prepare_model(training=True, test_code=False, load_weights=False, lr=0.001)
    #
    # finetune
    # finetune(ft_model_path='./model/squad_' + MODEL_NAME.replace('_ft', '') + '.hdf5')
    #
    # predict
    # pred = predict(text=False)
    # pd.to_pickle(pred, './pred.pkl')
