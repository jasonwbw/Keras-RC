from keras.layers.embeddings import Embedding
import os

import numpy as np
import pandas as pd
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from utils.data_utils import get_embedding_matrix
from model_zoo_for_MneReader import MneReader_Model

from utils.train_utils import Trainer_helper
from utils.evaluate_utils import Evaluate_helper
from utils.train_utils import *
# need self-generator

import sys
from configparser import ConfigParser

assert len(sys.argv) > 1 and '.cfg' in sys.argv[1], 'config file xxx.cfg should be given'

config_path = sys.argv[1]
cfg = ConfigParser()
cfg.read(config_path)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.get('Learning', 'GPU_CARD')

MODEL_NAME = cfg.get('Model', 'MODEL_NAME')

log_dir = './logs/'
output_name = './outputs/squad_' + MODEL_NAME + '.pkl'

GloveDimOption = cfg.getint('Hyper-parameters', 'EMBEDDING_DIM')
EMBEDDING_DIM = cfg.getint('Hyper-parameters', 'EMBEDDING_DIM')

BATCH_SIZE = cfg.getint('Learning', 'BATCH_SIZE')
MAX_SEQ_LEN = cfg.getint('Hyper-parameters', 'MAX_SEQ_LEN')

LEARNING_RATE = cfg.getfloat('Learning', 'LEARNING_RATE')
EARLY_STOP = cfg.getint('Learning', 'EARLY_STOP')
VALIDATION_SPLIT = cfg.getfloat('Learning', 'VALIDATION_SPLIT')
N_EPOCH = cfg.getint('Learning', 'N_EPOCH')


def load_data(mode='train'):
    X = pd.read_pickle('./input/' + mode + '_context_vec.pkl')
    X_char = pd.read_pickle('./input/' + mode + '_char_vec.pkl')
    Xq = pd.read_pickle('./input/' + mode + '_question_vec.pkl')
    Xq_char = pd.read_pickle('./input/' + mode + '_question_char_vec.pkl')
    embed_data = pd.read_pickle('./input/embed_features.pkl')
    if mode == 'train':
        Xq_tag = embed_data['pos_tag'][1][:len(X)]
        X_tag = embed_data['pos_tag'][0][:len(X)]
        Xq_ent = embed_data['entity'][1][:len(X)]
        X_ent = embed_data['entity'][0][:len(X)]
        Xq_em = embed_data['Extract_Match'][1][:len(X)]
        X_em = embed_data['Extract_Match'][0][:len(X)]
        Xq_type = embed_data['query_type'][:len(X)]
        XYBegin = pd.read_pickle('./input/' + mode + '_ybegin_vec.pkl')
        XYEnd = pd.read_pickle('./input/' + mode + '_yend_vec.pkl')
        return X, X_char, X_tag, X_ent, X_em, Xq, Xq_char, Xq_tag, Xq_ent, Xq_em, Xq_type, XYBegin, XYEnd

    len_q = len(pd.read_pickle('./input/' + 'train' + '_context_vec.pkl'))
    Xq_tag = embed_data['pos_tag'][1][len_q:]
    X_tag = embed_data['pos_tag'][0][len_q:]
    Xq_ent = embed_data['entity'][1][len_q:]
    X_ent = embed_data['entity'][0][len_q:]
    Xq_em = embed_data['Extract_Match'][1][len_q:]
    X_em = embed_data['Extract_Match'][0][len_q:]
    Xq_type = embed_data['query_type'][len_q:]
    return X, X_char, X_tag, X_ent, X_em, Xq, Xq_char, Xq_tag, Xq_ent, Xq_em, Xq_type


def train_valid_split(samples, VALIDATION_SPLIT=0.1):
    idx = np.arange(len(samples))
    np.random.seed(1024)
    np.random.shuffle(idx)
    idx_train = idx[:int(len(samples) * (1 - VALIDATION_SPLIT))]
    idx_val = idx[int(len(samples) * (1 - VALIDATION_SPLIT)):]
    return idx_train, idx_val


def combine_input_data(question_words, context_words, question_char, context_char, question_tag, context_tag,
                       question_ent, context_ent, question_em, context_em, question_type,
                       ids, mode='train'):
    ds = []
    ds.append(question_words)
    ds.append(question_char)
    ds.append(question_tag)
    ds.append(question_ent)
    ds.append(question_em)
    ds.append(question_type)
    ds.append(context_words)
    ds.append(context_char)
    ds.append(context_tag)
    ds.append(context_ent)
    ds.append(context_em)
    if mode == 'train':
        ds.append(ids)
    return ds


def batch_generator_v2(train, y, batch_size=128, shuffle=False, sort_by_length=False, group_by_length=True):
    sample_size = len(train[0])
    index_array = np.arange(sample_size)
    if group_by_length:
        # word legnth group not char level
        groups = calc_length_groups(sample_size, train, index=2)

    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        elif sort_by_length:
            index_array = np.argsort([-len(p) for p in train[1]])
        else:
            for k, v in groups.items():
                np.random.shuffle(v)
            tmp = np.concatenate(list(groups.values()))
            nb_batch = int(np.ceil(sample_size / float(batch_size)))
            split_batches = np.array_split(tmp, nb_batch)

            remainder = []
            if len(split_batches[-1]) < nb_batch:
                remainder = split_batches[-1:]
                split_batches = split_batches[:-1]
            np.random.shuffle(split_batches)
            split_batches += remainder
            index_array = np.concatenate(split_batches)

        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            X_batch_question_words = get_indcies_data(train[0], batch_ids)
            X_batch_question_char = get_indcies_data(train[1], batch_ids)
            X_batch_question_tag = get_indcies_data(train[2], batch_ids)
            X_batch_question_ent = get_indcies_data(train[3], batch_ids)
            X_batch_question_em = get_indcies_data(train[4], batch_ids)
            X_batch_question_type = get_indcies_data(train[5], batch_ids)

            X_batch_context_words = get_indcies_data(train[6], batch_ids)
            X_batch_context_char = get_indcies_data(train[7], batch_ids)
            X_batch_context_tag = get_indcies_data(train[8], batch_ids)
            X_batch_context_ent = get_indcies_data(train[9], batch_ids)
            X_batch_context_em = get_indcies_data(train[10], batch_ids)
            X_ids = get_indcies_data(train[11], batch_ids)

            Y_batch_st = get_indcies_data(y[0], batch_ids)
            Y_batch_ed = get_indcies_data(y[1], batch_ids)

            # padding
            X_batch_context_words = padded_batch_input(X_batch_context_words)
            word_len = X_batch_context_words.shape[1]
            X_batch_context_char = padded_batch_input(
                X_batch_context_char, maxlen=word_len)
            X_batch_context_tag = padded_batch_input(
                X_batch_context_tag, maxlen=word_len)
            X_batch_context_ent = padded_batch_input(
                X_batch_context_ent, maxlen=word_len)
            X_batch_context_em = padded_batch_input(
                X_batch_context_em, maxlen=word_len)
            X_batch_context_em = X_batch_context_em.reshape(
                X_batch_context_em.shape[0], X_batch_context_em.shape[1], 1)
            X_batch_context_em = X_batch_context_em.astype('float32')

            Y_batch_st = padded_batch_input(Y_batch_st, maxlen=word_len)
            Y_batch_ed = padded_batch_input(Y_batch_ed, maxlen=word_len)

            X_batch_question_words = padded_batch_input(X_batch_question_words)
            que_word_len = X_batch_question_words.shape[1]
            X_batch_question_char = padded_batch_input(
                X_batch_question_char, maxlen=que_word_len)
            X_batch_question_tag = padded_batch_input(
                X_batch_question_tag, maxlen=que_word_len)
            X_batch_question_ent = padded_batch_input(
                X_batch_question_ent, maxlen=que_word_len)
            X_batch_question_em = padded_batch_input(
                X_batch_question_em, maxlen=que_word_len)
            X_batch_question_em = X_batch_question_em.reshape(
                X_batch_question_em.shape[0], X_batch_question_em.shape[1], 1)
            X_batch_question_em = X_batch_question_em.astype('float32')
            X_batch_question_type = np.array(
                X_batch_question_type).reshape(-1, 1)

            X_batch = [X_batch_question_words, X_batch_question_char, X_batch_question_tag, X_batch_question_ent, X_batch_question_em, X_batch_question_type,
                       X_batch_context_words, X_batch_context_char, X_batch_context_tag, X_batch_context_ent, X_batch_context_em]

            yield X_batch, [Y_batch_st, Y_batch_ed], X_ids


def val_batch_generator_v2(train, batch_size=128):
    sample_size = len(train[0])
    index_array = np.arange(sample_size)
    while 1:
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            X_batch_question_words = get_indcies_data(train[0], batch_ids)
            X_batch_question_char = get_indcies_data(train[1], batch_ids)
            X_batch_question_tag = get_indcies_data(train[2], batch_ids)
            X_batch_question_ent = get_indcies_data(train[3], batch_ids)
            X_batch_question_em = get_indcies_data(train[4], batch_ids)
            X_batch_question_type = get_indcies_data(train[5], batch_ids)
            X_batch_context_words = get_indcies_data(train[6], batch_ids)
            X_batch_context_char = get_indcies_data(train[7], batch_ids)
            X_batch_context_tag = get_indcies_data(train[8], batch_ids)
            X_batch_context_ent = get_indcies_data(train[9], batch_ids)
            X_batch_context_em = get_indcies_data(train[10], batch_ids)

            # padding
            X_batch_context_words = padded_batch_input(X_batch_context_words)
            word_len = X_batch_context_words.shape[1]
            X_batch_context_char = padded_batch_input(
                X_batch_context_char, maxlen=word_len)
            X_batch_context_tag = padded_batch_input(
                X_batch_context_tag, maxlen=word_len)
            X_batch_context_ent = padded_batch_input(
                X_batch_context_ent, maxlen=word_len)
            X_batch_context_em = padded_batch_input(
                X_batch_context_em, maxlen=word_len)
            X_batch_context_em = X_batch_context_em.reshape(
                X_batch_context_em.shape[0], X_batch_context_em.shape[1], 1)
            X_batch_context_em = X_batch_context_em.astype('float32')

            X_batch_question_words = padded_batch_input(X_batch_question_words)
            que_word_len = X_batch_question_words.shape[1]
            X_batch_question_char = padded_batch_input(
                X_batch_question_char, maxlen=que_word_len)
            X_batch_question_tag = padded_batch_input(
                X_batch_question_tag, maxlen=que_word_len)
            X_batch_question_ent = padded_batch_input(
                X_batch_question_ent, maxlen=que_word_len)
            X_batch_question_em = padded_batch_input(
                X_batch_question_em, maxlen=que_word_len)
            X_batch_question_em = X_batch_question_em.reshape(
                X_batch_question_em.shape[0], X_batch_question_em.shape[1], 1)
            X_batch_question_em = X_batch_question_em.astype('float32')
            X_batch_question_type = np.array(
                X_batch_question_type).reshape(-1, 1)

            X_batch = [X_batch_question_words, X_batch_question_char, X_batch_question_tag, X_batch_question_ent, X_batch_question_em, X_batch_question_type,
                       X_batch_context_words, X_batch_context_char, X_batch_context_tag, X_batch_context_ent, X_batch_context_em]
            yield X_batch


def filter_longer(train_words, train_chars, train_tag, train_ent, train_em, train_q_words, train_q_chars,\
        train_q_tag, train_q_ent, train_q_em, train_q_type, train_y_begin, train_y_end, train_ids):
    lists = [train_words, train_chars, train_tag, train_ent, train_em, train_q_words, train_q_chars,\
        train_q_tag, train_q_ent, train_q_em, train_q_type, train_y_begin, train_y_end, train_ids]
    dels = []
    for i, x in enumerate(train_words):
        if len(x) > 300:
            dels.append(i)
    for i, iid in enumerate(dels):
        for j in range(len(lists)):
            del lists[j][iid - i]
    return lists



def prepare_model(training=True, test_code=False, load_weights=False, lr=0.0008, ft_model_path=None, epoch=None, adjust_lr=False, save_epoch=False, need_filter=False):
    train_words, train_chars, train_tag, train_ent, train_em, train_q_words, train_q_chars,\
        train_q_tag, train_q_ent, train_q_em, train_q_type, train_y_begin, train_y_end = load_data(
            'train')
    train_ids = pd.read_pickle('./input/train_question_id.pkl')
    if need_filter:
        filter_longer(train_words, train_chars, train_tag, train_ent, train_em, train_q_words, train_q_chars,\
            train_q_tag, train_q_ent, train_q_em, train_q_type, train_y_begin, train_y_end, train_ids)

    dev_words, dev_chars, dev_tag, dev_ent, dev_em,\
        dev_q_words, dev_q_chars, dev_q_tag, dev_q_ent, dev_q_em, dev_q_type = load_data(
            'dev')

    vocab2id = pd.read_pickle('./input/word_vocab.pkl')
    char2ids = pd.read_pickle('./input/char_vocab.pkl')
    tag2id = pd.read_pickle('./input/tag_dict.pkl')
    ent2id = pd.read_pickle('./input/ent_dict.pkl')

    embedding_matrix = get_embedding_matrix(vocab2id, EMBEDDING_DIM, GloveDimOption)

    train_data = combine_input_data(train_q_words, train_words, train_q_chars, train_chars, train_q_tag, train_tag, train_q_ent, train_ent,
                                    train_q_em, train_em, train_q_type, train_ids, 'train')
    train_y = [train_y_begin, train_y_end]
    dev_data = combine_input_data(dev_q_words, dev_words, dev_q_chars, dev_chars, dev_q_tag, dev_tag, dev_q_ent,
                                  dev_ent, dev_q_em, dev_em, dev_q_type, None, 'dev')

    # train model
    bst_model_path = './model/squad_' + MODEL_NAME + '.hdf5'
    if epoch is None:
        weights_path = bst_model_path
    else:
        weights_path = './model/squad_' + MODEL_NAME + '.ep%d.hdf5' % epoch

    # with tf.device('/cpu:0'):
    model = MneReader_Model(len(vocab2id), len(
        char2ids), embedding_matrix, len(tag2id), len(ent2id), cfg)
    # model = MneReader_Model(len(vocab2id), len(
    #     char2ids), embedding_matrix, 1000, 1000, use_highway=False)

    model.summary()
    # rms = optimizers.Adadelta(lr=(lr / 10) if training and load_weights else lr)  # default
    rms = optimizers.Adam(lr=(lr / 10) if training and load_weights else lr)  # default
    model.compile(optimizer=rms, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    evaluater = Evaluate_helper()
    trainer = Trainer_helper(model, log_dir='./logs/' + MODEL_NAME,
                             bst_model_path=bst_model_path, output_name=output_name, evaluate_helper=evaluater)
    trainer.set_train_generator(batch_generator_v2)
    trainer.set_valid_generator(val_batch_generator_v2)

    if load_weights:
        if training and ft_model_path is not None:
            trainer.load_weights(ft_model_path)
        else:
            trainer.load_weights(weights_path)

    if test_code:
        train_data = [d[:BATCH_SIZE * 2] for d in train_data]
        train_y = [d[:BATCH_SIZE * 2] for d in train_y]

    if training:
        trainer.fit(train_data, train_y, dev_data, batch_size=BATCH_SIZE, dev_batch_size=BATCH_SIZE//4, n_epoch=N_EPOCH, early_stop=EARLY_STOP,
                    verbose_train=100, is_save_intermediate=False, method=['group_by_length'], adjust_lr=adjust_lr, save_epoch=save_epoch)
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
    action = cfg.get('Action', 'ACTION')
    if action == 'train':
        # training
        prepare_model(training=True, test_code=False, load_weights=False, lr=LEARNING_RATE, adjust_lr=True, save_epoch=True)
    elif action == 'finetune':
        assert '_ft' in MODEL_NAME, 'finetune action should append the model name with _ft'
        # finetune
        finetune(ft_model_path='./model/squad_' + MODEL_NAME.replace('_ft', '') + '.hdf5')
    elif action == 'predict':
        # predict
        pred = predict(text=False)
        pd.to_pickle(pred, './pred.pkl')