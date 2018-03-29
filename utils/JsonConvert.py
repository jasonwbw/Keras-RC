import json
import pandas as pd

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


path = '../data/'

if __name__ == '__main__':
    dev_path = './data/dev-v1.1.json'
    data_entity_list = import_qas_data(dev_path)
    print('each paragraph context','---'*10)
    print(data_entity_list[0]._paragraphs_[0]._context_)
    print('each paragraph answer', '---' * 10)
    print('question id:',data_entity_list[0]._paragraphs_[0]._qas_[0]._id_)
    print('question:',data_entity_list[0]._paragraphs_[0]._qas_[0]._question_)
    print('answer start:',data_entity_list[0]._paragraphs_[0]._qas_[0]._answers_[0]._answer_start_)
    print('answer context:',data_entity_list[0]._paragraphs_[0]._qas_[0]._answers_[0]._text_)
    print('cur question answer numbers:',len(data_entity_list[0]._paragraphs_[0]._qas_[1]._answers_))
    print('cur paragraph question numbers:',len(data_entity_list[0]._paragraphs_[0]._qas_))

    pd.to_pickle(data_entity_list,'./data/dev_data.pkl')


    train_path = './data/train-v1.1.json'
    data_entity_list = import_qas_data(train_path)
    print(data_entity_list[0]._paragraphs_[0]._context_)
    print('each paragraph answer', '---' * 10)
    print('question id:', data_entity_list[0]._paragraphs_[0]._qas_[0]._id_)
    print('question:', data_entity_list[0]._paragraphs_[0]._qas_[0]._question_)
    print('answer start:', data_entity_list[0]._paragraphs_[0]._qas_[0]._answers_[0]._answer_start_)
    print('answer context:', data_entity_list[0]._paragraphs_[0]._qas_[0]._answers_[0]._text_)
    print('cur question answer numbers:', len(data_entity_list[0]._paragraphs_[0]._qas_[1]._answers_))
    print('cur paragraph question numbers:', len(data_entity_list[0]._paragraphs_[0]._qas_))

    pd.to_pickle(data_entity_list, './data/train_data.pkl')