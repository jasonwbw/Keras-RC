import numpy as np
import pandas as pd
import time
import datetime
from tqdm import tqdm
import nltk

import spacy
# from spacy.en import English
# parser  = English()
parser = spacy.load('en')


class Feature_helper(object):
    #indicate whether context words EM to query words
    def __init__(self,word2id,feature_name=None):
        if feature_name is None:
            feature_names = set(["Extract_Match",'pos_tag','entity','query_type'])
        self.feature_names = feature_names
        self.feature_extrator = {}
        self.id2words = {i:k for k,i in word2id.items()}

        key = "Extract_Match"
        if key in feature_names:
            self.feature_extrator[key] = self.get_extract_match_embed

        key = 'pos_tag'
        if key in feature_names:
            self.feature_extrator[key] = self.get_pos_tag_embed

        key = 'entity'
        if key in feature_names:
            self.feature_extrator[key] = self.get_entity_embed

        key ='query_type'
        if key in feature_names:
            self.feature_extrator[key] =self.get_query_type_embed


    def fit(self,contexts_str,questions_str,is_load=False):
        self.parse_q1 = {}
        self.parse_q2 = {}
        for i in tqdm(np.arange(len(contexts_str))):
            self.parse_q1[i] = parser(str(contexts_str[i]))
            self.parse_q2[i] = parser(str(questions_str[i]))
        if is_load:
            self.tag_dict = pd.read_pickle('./input/tag_dict.pkl')
            self.ent_dict = pd.read_pickle('./input/ent_dict.pkl')
        else:
            tag_dict = self.generate_tag_dict()
            self.tag_dict = tag_dict
            ent_dict = self.generate_entity_dict()
            self.ent_dict = ent_dict

    def generate_tag_dict(self):
        tag_dict = {}
        for i in tqdm(np.arange(len(self.parse_q1))):
            parse_1 = self.parse_q1[i]
            parse_2 = self.parse_q2[i]
            for token in parse_1:
                tag = str(token.pos_)
                if (tag not in tag_dict):
                    tag_dict[tag] = 1
                else:
                    tag_dict[tag]+=1
            for token in parse_2:
                tag = str(token.pos_)
                if (tag not in tag_dict):
                    tag_dict[tag] = 1
                else:
                    tag_dict[tag]+=1
        tag_dict = {k: i + 1 for i, k in enumerate(tag_dict)}
        return tag_dict


    def generate_entity_dict(self):
        ent_dict = {}
        for i in tqdm(np.arange(len(self.parse_q1))):
            parse_1 = self.parse_q1[i]
            parse_2 = self.parse_q2[i]
            for token in parse_1:
                tag = str(token.ent_type)
                if (tag not in ent_dict):
                    ent_dict[tag] = 1
                else:
                    ent_dict[tag]+=1
            for token in parse_2:
                tag = str(token.ent_type)
                if (tag not in ent_dict):
                    ent_dict[tag] = 1
                else:
                    ent_dict[tag]+=1
        ent_dict = {k:i+1 for i,k in enumerate(ent_dict)}
        return ent_dict

    def get_pos_tag_embed(self):
        con_features = []
        que_features = []
        for i in tqdm(np.arange(len(self.parse_q1))):
            parse_1 = self.parse_q1[i]
            parse_2 = self.parse_q2[i]
            tagids = [self.tag_dict[str(token.pos_)] if str(token.pos_) in self.tag_dict else 0  for token in parse_1]
            tagids2 = [self.tag_dict[str(token.pos_)] if str(token.pos_) in self.tag_dict else 0  for token in parse_2]
            con_features.append(tagids)
            que_features.append(tagids2)
        return con_features,que_features

    def get_entity_embed(self):
        con_features = []
        que_features = []
        for i in tqdm(np.arange(len(self.parse_q1))):
            parse_1 = self.parse_q1[i]
            parse_2 = self.parse_q2[i]
            entids = [self.ent_dict[str(token.ent_type)] if str(token.ent_type) in self.ent_dict else 0  for token in parse_1]
            entids2 = [self.ent_dict[str(token.ent_type)] if str(token.ent_type) in self.ent_dict else 0  for token in parse_2]
            con_features.append(entids)
            que_features.append(entids2)
        return con_features, que_features

    def get_extract_match_embed(self):
        samples = len(self.parse_q1)
        con_features = []
        que_features = []
        for ind in range(samples):
            context = [str(token) for token in self.parse_q1[ind]]
            question = [str(token) for token in self.parse_q2[ind]]
            con_fea = [1 if cw in question else 0 for cw in context]
            que_fea = [1 if cw in context else 0 for cw in question]
            con_features.append(con_fea)
            que_features.append(que_fea)
        return con_features,que_features

    def get_query_type_embed(self):
        samples = len(self.parse_q2)
        que_features = []
        query_type = ['what','how','who','when','which','where','why','be']
        query_type_dict = {k:i+1 for i,k in enumerate(query_type)}
        for ind in range(samples):
            question = [str(token) for token in self.parse_q2[ind]]
            cur_type = [query_type_dict[qu_type] for qu_type in query_type if qu_type in question]
            if len(cur_type)==0:
                que_features.append(0)
            else:
                que_features.append(cur_type[0])
        return que_features

    def transform(self):
        data = {}
        for key in self.feature_names:
            if key in self.feature_extrator:
                data[key] = self.feature_extrator[key]()
        return data

    def save(self):
        pd.to_pickle(self.tag_dict,'../input/tag_dict.pkl')
        pd.to_pickle(self.ent_dict,'../input/ent_dict.pkl')


def generate_corpus(words):
    ret = []
    for ind in tqdm(range(len(words))):
        ret.append(' '.join([id2words[cw] for cw in words[ind]]))
    return  ret


def load_data(mode='train'):
    X = pd.read_pickle('../input/' + mode + '_context_vec.pkl')
    X_char = pd.read_pickle('../input/' + mode + '_char_vec.pkl')
    Xq = pd.read_pickle('../input/' + mode + '_question_vec.pkl')
    Xq_char = pd.read_pickle('../input/' + mode + '_question_char_vec.pkl')
    if mode == 'train':
        XYBegin = pd.read_pickle('../input/' + mode + '_ybegin_vec.pkl')
        XYEnd = pd.read_pickle('../input/' + mode + '_yend_vec.pkl')
        return X, X_char, Xq, Xq_char, XYBegin, XYEnd
    return X, X_char, Xq, Xq_char


if __name__ == '__main__':
    train_words, train_chars, train_q_words, train_q_chars, train_y_begin, train_y_end = load_data('train')
    train_ids = pd.read_pickle('../input/train_question_id.pkl')
    dev_words, dev_chars, dev_q_words, dev_q_chars = load_data('dev')
    vocab2id = pd.read_pickle('../input/word_vocab.pkl')

    id2words = {i: k for k, i in vocab2id.items()}
    context_corpus = generate_corpus(train_words + dev_words)
    que_corpus = generate_corpus(train_q_words + dev_q_words)

    feature_extrator = Feature_helper(vocab2id,None)
    feature_extrator.fit(context_corpus,que_corpus,False)
    feature_extrator.save()
    features = feature_extrator.transform()    

    pd.to_pickle(features,'../input/embed_features.pkl')






