import os
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from tqdm import tqdm
import pandas as pd
import sys
from .evaluate_utils import Evaluate_helper


def lengthGroup(length):
    '''
    for the new data
    '''
    return length // 50

# def lengthGroup(length):
#     if length < 150:
#         return 0
#     if length < 240:
#         return 1
#     if length < 380:
#         return 2
#     if length < 520:
#         return 3
#     if length < 660:
#         return 4
#     return 5

# train on batch


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]


import itertools


def calc_length_groups(nb_samples, inputs, index=2):  # index denote word level
    indices = np.arange(nb_samples)

    def ff(i): return lengthGroup(len(inputs[index][i]))

    indices = np.argsort([ff(i) for i in indices])

    groups = itertools.groupby(indices, ff)

    groups = {k: np.array(list(v)) for k, v in groups}
    return groups


def padded_batch_input(input, indices=None, maxlen=None):
    if indices is None:
        indices = np.arange(len(input))
    batch_ = [input[i] for i in indices]
    return sequence.pad_sequences(batch_, maxlen, padding='post')


def get_indcies_data(data, indices=None):
    if indices is None:
        indices = np.arange(len(data))
    return [data[i] for i in indices]


def batch_generator(train, y, batch_size=128, shuffle=False, sort_by_length=False, group_by_length=True):
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
            X_batch_context_words = get_indcies_data(train[2], batch_ids)
            X_batch_context_char = get_indcies_data(train[3], batch_ids)
            X_ids = get_indcies_data(train[4], batch_ids)

            Y_batch_st = get_indcies_data(y[0], batch_ids)
            Y_batch_ed = get_indcies_data(y[1], batch_ids)

            # padding
            X_batch_context_words = padded_batch_input(X_batch_context_words)
            word_len = X_batch_context_words.shape[1]
            X_batch_context_char = padded_batch_input(
                X_batch_context_char, maxlen=word_len)
            Y_batch_st = padded_batch_input(Y_batch_st, maxlen=word_len)
            Y_batch_ed = padded_batch_input(Y_batch_ed, maxlen=word_len)

            X_batch_question_words = padded_batch_input(X_batch_question_words)
            que_word_len = X_batch_question_words.shape[1]
            X_batch_question_char = padded_batch_input(
                X_batch_question_char, maxlen=que_word_len)

            X_batch = [X_batch_question_words, X_batch_question_char,
                       X_batch_context_words, X_batch_context_char]

            yield X_batch, [Y_batch_st, Y_batch_ed], X_ids


def val_batch_generator(train, batch_size=128):
    sample_size = len(train[0])
    index_array = np.arange(sample_size)
    while 1:
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            X_batch_question_words = get_indcies_data(train[0], batch_ids)
            X_batch_question_char = get_indcies_data(train[1], batch_ids)
            X_batch_context_words = get_indcies_data(train[2], batch_ids)
            X_batch_context_char = get_indcies_data(train[3], batch_ids)

            # padding
            X_batch_context_words = padded_batch_input(X_batch_context_words)
            word_len = X_batch_context_words.shape[1]
            X_batch_context_char = padded_batch_input(
                X_batch_context_char, maxlen=word_len)
            X_batch_question_words = padded_batch_input(X_batch_question_words)
            que_word_len = X_batch_question_words.shape[1]
            X_batch_question_char = padded_batch_input(
                X_batch_question_char, maxlen=que_word_len)

            X_batch = [X_batch_question_words, X_batch_question_char,
                       X_batch_context_words, X_batch_context_char]
            yield X_batch


class Trainer_helper(object):
    def __init__(self, model, log_dir='./logs/', bst_model_path='./model/squad_rnet.hdf5', output_name='./', evaluate_helper=None):
        """
        :param model:
        :param log_dir:  log path
        :param bst_model_path: model path
        :param output_name: prediction save path
        :param evaluate_helper:  evaluater helper for evaluate train on batch and dev data
        """
        self.model = model
        if os.path.exists(log_dir) == False:
            os.mkdir(log_dir)
        self.bst_model_path = bst_model_path
        self.output_name = output_name
        self.output_name_json = output_name[:-3] + 'json'
        self.tf_train_writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'train'))
        self.tf_test_writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'dev'))
        self.is_save_intemediate = False
        self.train_groud_truth = pd.read_pickle('./input/train_id2answer.pkl')
        self.set_train_generator()  # default
        self.set_valid_generator()  # default
        if evaluate_helper is None:
            self.evaluate_helper = Evaluate_helper()
        else:
            self.evaluate_helper = evaluate_helper

    def load_weights(self, model_path):
        self.model.load_weights(model_path)

    def set_train_generator(self, gen=None):
        if gen is not None:
            self.tr_gen = gen
        else:
            self.tr_gen = batch_generator

    def set_valid_generator(self, gen=None):
        if gen is not None:
            self.te_gen = gen
        else:
            self.te_gen = val_batch_generator

    def save_logs(self, em, f1, i, mode='train'):
        logs = {'Extract_Match': em, 'F1': f1}
        for name, value in logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            if mode == 'train':
                self.tf_train_writer.add_summary(summary, i)
            else:
                self.tf_test_writer.add_summary(summary, i)

        if mode == 'train':
            self.tf_train_writer.flush()
        else:
            self.tf_test_writer.flush()

    def fit(self, train_x, train_y, valid_x,
            n_epoch, batch_size=1, dev_batch_size=10, early_stop=1, verbose_train=200, is_save_intermediate=False,
            save_epoch=False, adjust_lr=False,
            method=['shuffle', 'sort_by_length', 'group_by_length']):
        """
        :param train_x:
        :param train_y:
        :param valid_x:
        :param n_epoch:
        :param early_stop:
        :param batch_size:
        :param is_save_intermediate: is save intermediate prediction prob
        :param method: train method
        :param metrics: Extract match or F1 score
        :return:
        """
        self.is_save_intemediate = is_save_intermediate
        early_stop_step = 0
        best_em = 0
        best_f1 = 0
        is_flag = 0
        if 'shuffle' in method:
            is_flag = 1
        if 'sort_by_length' in method:
            is_flag = 2
        if 'group_by_length' in method:
            is_flag = 3

        for i in tqdm(range(n_epoch), desc="Epoch Completed"):
            if is_flag == 3:
                tr_gen = self.tr_gen(train_x, train_y, batch_size=batch_size,
                                     shuffle=False, sort_by_length=False, group_by_length=True)
            elif is_flag == 2:
                tr_gen = self.tr_gen(train_x, train_y, batch_size=batch_size,
                                     shuffle=False, sort_by_length=True, group_by_length=False)
            else:
                tr_gen = self.tr_gen(train_x, train_y, batch_size=batch_size,
                                     shuffle=True, sort_by_length=False, group_by_length=False)

            loss = 0
            answer_start_loss = 0
            answer_end_loss = 0
            answer_start_acc = 0
            answer_end_acc = 0
            cnt = 0
            cnt_tr = 0
            em_tr = 0
            f1_tr = 0
            steps_per_epoch = int(np.ceil(len(train_x[0]) / float(batch_size)))
            # self.model.fit_generator(tr_gen, steps_per_epoch=steps_per_epoch,epochs=1, verbose=1, max_q_size=20)
            for (X_batch, Y_batch, X_ids), step in tqdm(zip(tr_gen, range(steps_per_epoch)), total=steps_per_epoch, ncols=50, leave=False):
                val = self.model.train_on_batch(X_batch, Y_batch, )
                element_cnt = X_batch[0].shape[0]
                loss += val[0] * element_cnt
                answer_start_loss += val[1] * element_cnt
                answer_end_loss += val[2] * element_cnt
                answer_start_acc += val[3] * element_cnt
                answer_end_acc += val[4] * element_cnt
                cnt += element_cnt
                print('----train loss:%f,answer start loss:%f,answer end loss:%f,answer start acc:%f,answer end acc:%f-------' % (loss / cnt,
                                                                                                                                  answer_start_loss / cnt,
                                                                                                                                  answer_end_loss / cnt,
                                                                                                                                  answer_start_acc / cnt,
                                                                                                                                  answer_end_acc / cnt))
                if step % verbose_train == 0:
                    predictions = self.model.predict(X_batch)
                    answer = self.evaluate_helper.dump_answer(
                        predictions, X_ids, mode='train')
                    t_em, t_f1 = self.evaluate_helper.evaluate_train(
                        answer, self.train_groud_truth)
                    em_tr += t_em * element_cnt
                    f1_tr += t_f1 * element_cnt
                    cnt_tr += element_cnt
                    print('--train Extract Match score :%f--------F1 score:%f' %
                          (em_tr / cnt_tr, f1_tr / cnt_tr))
                    self.save_logs(em_tr / cnt_tr, f1_tr / cnt_tr, i *
                                   steps_per_epoch + step, mode='train')  # total step

                sys.stdout.flush()

            em, f1 = self.evaluate_on_dev(
                valid_x, dev_batch_size, i)  # can control huge data
            print('--Dev Extract Match score :%f--------F1 score:%f' % (em, f1))

            self.save_logs(em, f1, i, mode='dev')
            if em > best_em or f1 > best_f1:
                self.model.save_weights(self.bst_model_path)
                print('save model to ', self.bst_model_path)
                if save_epoch:
                    self.model.save_weights(
                        self.bst_model_path.replace('.hdf5', '.ep%d.hdf5' % i))
                    print('save model to ', self.bst_model_path.replace(
                        '.hdf5', '.ep%d.hdf5' % i))
                if em > best_em:
                    best_em = em
                if f1 > best_f1:
                    best_f1 = f1
                early_stop_step = 0
            else:
                early_stop_step += 1
                if early_stop_step >= early_stop:
                    print('early stop @', i)
                    break
                if adjust_lr:
                    lr = float(K.get_value(self.model.optimizer.lr))
                    K.set_value(self.model.optimizer.lr, lr / 2)

    def get_predict(self, valid_x, batch_size):
        te_gen = self.te_gen(valid_x, batch_size=batch_size)
        steps = int(np.ceil(len(valid_x[0]) / float(batch_size)))
        st = []
        ed = []
        for X, step in tqdm(zip(te_gen, range(steps)), total=steps):
            # batch on predict acclerate betten on predict
            _st, _ed = self.model.predict_on_batch(X)
            st.append(_st)
            ed.append(_ed)
        return [st, ed]

    def evaluate_on_dev(self, valid_x, batch_size, epoch_idx):
        predictions = self.get_predict(valid_x, batch_size)
        if self.is_save_intemediate:
            pd.to_pickle(predictions, epoch_idx + '_' + self.output_name)
        dev_ids = pd.read_pickle('./input/dev_question_id.pkl')
        answers = self.evaluate_helper.dump_answer(
            predictions, dev_ids, mode='dev')
        return self.evaluate_helper.evaluate_dev(answers)

    def predict_probs(self, valid_x, batch_size):
        predictions = self.get_predict(valid_x, 10)
        return predictions

    def predict(self, valid_x, batch_size):
        predictions = self.get_predict(valid_x, 10)
        dev_ids = pd.read_pickle('./input/dev_question_id.pkl')
        answers = self.evaluate_helper.dump_answer(
            predictions, dev_ids, mode='dev')
        return answers
