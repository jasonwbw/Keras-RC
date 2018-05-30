import numpy as np
import pandas as pd
import re
import json
import os
# from pprint import pprint
# import nltk
from collections import Counter
from tqdm import tqdm
import spacy

from keras.preprocessing.sequence import pad_sequences


nlp = spacy.load('en')

seed = 1024
np.random.seed(seed)


def customized_tokenizer(sent):
    return nlp.tokenizer(sent)


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    # return [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sent)]
    return [str(token).replace("``", '"').replace("''", '"') for token in customized_tokenizer(sent)]


class Answer:
    def __init__(self, answer_start, text):
        self._answer_start_ = answer_start
        self._text_ = text


class QAEntity:
    def __init__(self, question, id):
        self._question_ = question
        self._id_ = id
        self._answers_ = []


class Paragraph:
    def __init__(self, context, qas):
        self._context_ = context
        self._qas_ = []
        for answer in qas:
            qa = QAEntity(answer['question'], answer['id'])
            for item in answer['answers']:
                a = Answer(item['answer_start'], item['text'])
                qa._answers_.append(a)
            self._qas_.append(qa)


class DataEntity:
    def __init__(self, title, paragraph_data):
        self._title_ = title
        self._paragraphs_ = []
        for item in paragraph_data:
            paragraph = Paragraph(item['context'], item['qas'])
            self._paragraphs_.append(paragraph)


def import_qas_data(datapath):
    with open(datapath) as data_file:
        data = json.load(data_file)

    data_entity_list = []
    for item in data['data']:
        entity = DataEntity(item['title'], item['paragraphs'])
        data_entity_list.append(entity)
    return data_entity_list


# only word_char_level need paddingg
configs = {
    "word_count_th": 5,
    'char_count_th': 20,
    'sent_len_th': 767,
    'ques_len_th': 50,
    'word_char_size_th': 16,
}


def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'",
            "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens


def cout_sentences(data_entity_list):
    separators = ['.', '?', '!']
    lens = Counter()
    for data in tqdm(data_entity_list):
        paragraphs = data._paragraphs_
        for paragraph in paragraphs:
            context = paragraph._context_
            context1 = context.replace("''", '" ')
            context1 = context1.replace("``", '" ')
            context1 = ' '.join(context1.split(" ")[:300])
            count = 0
            for sep in separators:
                count += context1.count(sep)
            lens[count] += 1
            if count > 40:
                print(context1)
            
            # qas = paragraph._qas_
            # for qa in qas:
            #     question = qa._question_
            #     question = question.replace("''", '" ')
            #     question = question.replace("``", '" ')

            #     question_id = qa._id_
            #     answers = qa._answers_
            #     for answer in answers:
            #         answerText = answer._text_
    sum_of_numbers = sum(l * lens[l] for l in lens.elements())
    count = sum(lens[l] for l in lens.elements())
    mean = sum_of_numbers / count
    print(mean)
    print(lens.most_common(len(lens)))


def count_q_doc_overlaps(data_entity_list):
    import re
    separators = ['.', '?', '!']
    overlap_rates, overlaps, count = 0., 0, 0
    for data in tqdm(data_entity_list):
        paragraphs = data._paragraphs_
        for paragraph in paragraphs:
            context = paragraph._context_
            context1 = context.replace("''", '" ')
            context1 = context1.replace("``", '" ')
            context1 = ' '.join(context1.split(" ")[:300])
            sentences = re.split("[\.\?!]+", context1)
            # print(context1)
            # print(sentences)
            # break
            
            qas = paragraph._qas_
            for qa in qas:
                question = qa._question_
                question = question.replace("''", '" ')
                question = question.replace("``", '" ')
                question_ws = set(question.split(' '))

                question_id = qa._id_
                answers = qa._answers_
                for answer in answers:
                    answerText = answer._text_
                    contain_ids = []
                    other_ids = []
                    for i, s in enumerate(sentences):
                        if answerText in s:
                            contain_ids.append(i)
                        else:
                            other_ids.append(i)
                    overlap = 0
                    for i in other_ids:
                        s_ws = set(sentences[i].split(' '))
                        # print(question_ws)
                        # print(s_ws)
                        # print(answerText)
                        # print(question_ws & s_ws)
                        # print()
                        overlap = max(overlap, len(question_ws & s_ws))
                    overlap_rates += float(overlap) / len(question_ws)
                    overlaps += overlap
                    count += 1
        # break

    print(overlap_rates / count)
    print(overlaps / count)


def split_data(data_entity_list):
    '''Given a parsed Json data object, split the object into training context (paragraph), question, answer matrices,
           and keep track of max context and question lengths.
    '''
    xContext = []  # list of contexts paragraphs
    xContext_char = []
    xQuestion = []  # list of questions
    xQuestion_char = []
    xQuestion_id = []  # list of question id
    xAnswerBegin = []  # list of indices of the beginning word in each answer span
    xAnswerEnd = []  # list of indices of the ending word in each answer span
    xAnswerText = []  # list of the answer text
    maxSeqLen = 0
    maxSeqQuestionLen = 0

    for data in tqdm(data_entity_list):
        paragraphs = data._paragraphs_
        for paragraph in paragraphs:
            context = paragraph._context_
            context1 = context.replace("''", '" ')
            context1 = context1.replace("``", '" ')
            contextTokenized = tokenize(context1.lower())
            contextTokenized_char = [list(xij) for xij in contextTokenized]
            paraLen = len(contextTokenized)
            if paraLen > maxSeqLen:
                maxSeqLen = paraLen

            qas = paragraph._qas_
            for qa in qas:
                question = qa._question_
                question = question.replace("''", '" ')
                question = question.replace("``", '" ')
                questionTokenized = tokenize(question.lower())
                cquestionTokenized = [list(qij) for qij in questionTokenized]

                if len(questionTokenized) > maxSeqQuestionLen:
                    maxSeqQuestionLen = len(questionTokenized)
                question_id = qa._id_
                answers = qa._answers_
                for answer in answers:
                    answerText = answer._text_
                    answerTokenized = tokenize(answerText.lower())
                    # find indices of beginning/ending words of answer span among tokenized context
                    contextToAnswerFirstWord = context1[:answer._answer_start_ + len(
                        answerTokenized[0])]
                    answerBeginIndex = len(
                        tokenize(contextToAnswerFirstWord.lower())) - 1
                    answerEndIndex = answerBeginIndex + \
                        len(answerTokenized) - 1

                    xContext.append(contextTokenized)
                    xContext_char.append(contextTokenized_char)
                    xQuestion.append(questionTokenized)
                    xQuestion_char.append(cquestionTokenized)
                    xQuestion_id.append(str(question_id))
                    xAnswerBegin.append(answerBeginIndex)
                    xAnswerEnd.append(answerEndIndex)
                    xAnswerText.append(answerText)

    return xContext, xContext_char, xQuestion, xQuestion_char, xQuestion_id, xAnswerBegin, \
        xAnswerEnd, xAnswerText, maxSeqLen, maxSeqQuestionLen


def tokenizeVal(sent):
    tokenizedSent = tokenize(sent)
    tokenIdx2CharIdx = [None] * len(tokenizedSent)
    idx = 0
    token_idx = 0
    while idx < len(sent) and token_idx < len(tokenizedSent):
        word = tokenizedSent[token_idx]  # each word
        # word_num_formated = format(int(word), ',') if word.isdigit() else None
        if sent[idx:idx + len(word)] == word:
            tokenIdx2CharIdx[token_idx] = idx
            idx += len(word)
            token_idx += 1
        # elif word_num_formated is not None and sent[idx:idx + len(word_num_formated)] == word_num_formated:
        #     tokenIdx2CharIdx[token_idx] = idx
        #     idx += len(word_num_formated)
        #     token_idx += 1
        else:
            idx += 1
    for t in tokenIdx2CharIdx:
        assert t is not None, 'all the tokenized words should be mapped to the original text. \n[origin] %s; \n[tokenized] %s.\n[%s]' % (sent, tokenizedSent, ', '.join(list(map(str, tokenIdx2CharIdx, ))))
    return tokenizedSent, tokenIdx2CharIdx


def read_train_data(data_entity_list, pre_number, post_number):
    '''Given a parsed Json data object, split the object into training context (paragraph), question, answer matrices,
           and keep track of max context and question lengths.
    '''
    xContext = []  # list of contexts paragraphs
    xContext_char = []
    xQuestion = []  # list of questions
    xQuestion_char = []
    xQuestion_id = []  # list of question id
    xToken2CharIdx = []
    xContextOriginal = []

    for data in tqdm(data_entity_list):
        paragraphs = data._paragraphs_
        for paragraph in paragraphs:
            context = paragraph._context_
            context1 = context.replace("''", '" ')
            context1 = context1.replace("``", '" ')
            contextTokenized, tokenIdx2CharIdx = tokenizeVal(context1.lower())
            contextTokenized_char = [list(xij) for xij in contextTokenized]
            qas = paragraph._qas_
            for qa in qas:
                question = qa._question_
                question = question.replace("''", '" ')
                question = question.replace("``", '" ')
                questionTokenized = tokenize(question.lower())
                questionTokenized_char = [list(xij)
                                          for xij in questionTokenized]
                question_id = qa._id_
                answers = qa._answers_
                for answer in answers:
                    # answerText = answer._text_
                    # answerTokenized = tokenize(answerText.lower())
                    xToken2CharIdx.append(tokenIdx2CharIdx)
                    xContextOriginal.append(context)
                    xContext.append(contextTokenized)
                    xQuestion.append(questionTokenized)
                    xContext_char.append(contextTokenized_char)
                    xQuestion_char.append(questionTokenized_char)
                    xQuestion_id.append(str(question_id))

    return xContext, xContext_char, xToken2CharIdx, xContextOriginal, xQuestion, xQuestion_char, xQuestion_id


# from collections import Counter


def create_vocab(corpus):
    UNK = "-UNK-"
    NULL = "-NULL-"
    vocab = Counter()
    char_vocab = Counter()
    for sent in tqdm(corpus):
        for word in sent:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
            for ch in list(word):
                if ch not in char_vocab:
                    char_vocab[ch] = 1
                else:
                    char_vocab[ch] += 1
    vocab = [v[0] for v in sorted(vocab.items(
    ), key=lambda x:x[1], reverse=True) if v[1] >= configs['word_count_th']]
    char_vocab = [v[0] for v in sorted(char_vocab.items(
    ), key=lambda x:x[1], reverse=True) if v[1] >= configs['char_count_th']]
    vocab_size = len(vocab) + 2
    vocab2id = dict((c, i + 2) for i, c in enumerate(vocab))
    char_vocab_size = len(char_vocab) + 2
    charvocab2id = dict((c, i + 2) for i, c in enumerate(char_vocab))
    vocab2id[UNK] = 1
    charvocab2id[UNK] = 1
    vocab2id[NULL] = 0
    charvocab2id[NULL] = 0
    return vocab_size, vocab2id, char_vocab_size, charvocab2id


# can use mask to filter num sentences


def vectorizeData(xContext, xContext_char,
                  xQuestion, xQuestion_char,
                  xAnswerBeing, xAnswerEnd, keep_length=True):
    '''Vectorize the words to their respective index and pad context to max context length and question to max question length.
       Answers vectors are padded to the max context length as well.
    '''
    X = []
    X_char = []
    Xq = []
    Xq_char = []
    YBegin = []
    YEnd = []
    for i in tqdm(range(len(xContext))):
        x = [vocab2id[w] if w in vocab2id else 1 for w in xContext[i]]
        x_char = [[char2ids[c] if c in char2ids else 1 for c in w]
                  for w in xContext_char[i]]

        xq = [vocab2id[w] if w in vocab2id else 1 for w in xQuestion[i]]
        xq_char = [[char2ids[c] if c in char2ids else 1 for c in w]
                   for w in xQuestion_char[i]]
        # map the first and last words of answer span to one-hot representations
        y_Begin = np.zeros(len(xContext[i]))  # boundary how many words
        y_Begin[xAnswerBeing[i]] = 1
        y_End = np.zeros(len(xContext[i]))
        y_End[xAnswerEnd[i]] = 1
        X.append(x)
        X_char.append(x_char)
        Xq.append(xq)
        Xq_char.append(xq_char)
        YBegin.append(y_Begin)
        YEnd.append(y_End)
    print('vectorized end...')
    print('begin padding...')
    # padding
    # only padding char
    for i in range(len(X_char)):
        X_char[i] = pad_sequences(
            X_char[i], maxlen=configs['word_char_size_th'], padding='post')
    for i in range(len(Xq)):
        Xq_char[i] = pad_sequences(
            Xq_char[i], maxlen=configs['word_char_size_th'], padding='post')

    if keep_length is not True:
        X = pad_sequences(X, maxlen=configs['sent_len_th'], padding='post')
        X_char = pad_sequences(
            X_char, maxlen=configs['sent_len_th'], padding='post')
        Xq = pad_sequences(Xq, maxlen=configs['ques_len_th'], padding='post')
        Xq_char = pad_sequences(
            Xq_char, maxlen=configs['ques_len_th'], padding='post')
        YBegin = pad_sequences(
            YBegin, maxlen=configs['sent_len_th'], padding='post')
        YEnd = pad_sequences(
            YEnd, maxlen=configs['sent_len_th'], padding='post')

    return X, X_char, Xq, Xq_char, YBegin, YEnd


def save_data(X, X_char, Xq, Xq_char, XYBegin=None, XYEnd=None, mode='train'):
    pd.to_pickle(X, './input/' + mode + '_context_vec.pkl')
    pd.to_pickle(X_char, './input/' + mode + '_char_vec.pkl')
    pd.to_pickle(Xq, './input/' + mode + '_question_vec.pkl')
    pd.to_pickle(Xq_char, './input/' + mode + '_question_char_vec.pkl')
    if mode == 'train':
        pd.to_pickle(XYBegin, './input/' + mode + '_ybegin_vec.pkl')
        pd.to_pickle(XYEnd, './input/' + mode + '_yend_vec.pkl')



def map_question_id_to_data(question_ids, data):
    return {id: c for id, c in zip(question_ids, data)}


def mod_long_context(context, begin_loc, end_loc, special_flag=False):
    sep_sent_symbol = ['.', '!', '?']
    if special_flag:
        sep_sent_symbol.append(',')
    sp_sent = []
    start = 0
    for j in range(len(context)):
        if context[j] in sep_sent_symbol:
            sp_sent.append(context[start:j+1])
            start = j+1
        else:
            continue
    else:
        if start != len(context):
            sp_sent.append(context[start:])
    sp_length = [len(sent) for sent in sp_sent]
    sp_begin_index = find_index(sp_length, begin_loc)
    sp_end_index = find_index(sp_length, end_loc)
    total = sum(sp_length[sp_begin_index:sp_end_index+1])
    start = sp_begin_index
    end = sp_end_index

    # if param context length smaller than th, infinite loop.
    while total < length_th:
        if start > 0:
            if total + sp_length[start-1] > length_th:
                break
            else:
                start -= 1
                total += sp_length[start]
        if end < len(sp_length)-1:
            if total + sp_length[end+1] > length_th:
                break
            else:
                end += 1
                total += sp_length[end]

    return_sent = []
    for x in sp_sent[start:end+1]:
        return_sent += x
    save_length = len(return_sent)
    _before = sum(sp_length[:start])
    _post = sum(sp_length[end+1:])
    # not onehot, just scalar
    # mod_begin = convert_index(begin_loc, save_length, _before)
    # mod_end = convert_index(end_loc, save_length, _before)
    mod_begin = begin_loc - _before
    mod_end = end_loc - _before

    return return_sent, mod_begin, mod_end, _before, _post


def find_index(sp_length, loc):
    total = 0
    for i in range(len(sp_length)):
        if total+sp_length[i] < loc:
            total += sp_length[i]
        else:
            return i


def convert_index(loc, length, _before):
    loc = loc - _before
    new_index = np.zeros(length)
    new_index[loc] = 1

    return new_index


def print_context_length(context):
    count = dict()
    for line in tqdm(context):
        _length = count.get(len(line))
        if _length is None:
            count[len(line)] = 1
        else:
            count[len(line)] += 1
    sort_count = sorted(count.items(), key=lambda x: x[0])
    print(sort_count)


if __name__ == '__main__':
    if not os.path.exists('./input/'):
        os.mkdir('./input/')
    print('json data convert', '-' * 30)
    train_path = './data/train-v1.1.json'
    train_entity_list = import_qas_data(train_path)
    dev_path = './data/dev-v1.1.json'
    dev_entity_list = import_qas_data(dev_path)

    # cout_sentences(train_entity_list)
    count_q_doc_overlaps(train_entity_list)
