from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, InputLayer, add
from keras.layers.core import Dense, RepeatVector, Masking, Dropout
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers import recurrent, Dense, Input, Dropout, TimeDistributed, Flatten, concatenate, Embedding, Merge, merge, multiply
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from utils.layers.attention_layer4Mnemonic import Interactive_Align_attention, Self_Align_attention, MemoryBasedPointer
from utils.layers.core_layer import Slice, SharedWeight

max_char_size = 16
# EMBEDDING_DIM = 100
CHAR_EMBEDDING_DIM = 50
CHAR_EMBED_METHOD = 'RNN'  # CNN
HIDDEN_SIZE = 100
MAX_SEQ_LEN = 400
ENCODER_LAYERS = 1
RNN_Cell = 'LSTM'
DP = 0.2
UNROLL = False
MLP_LAYER = 1

HOPS = 2


def get_share_weight_():
    Wci = SharedWeight(size=(2 * HIDDEN_SIZE, 2 * HIDDEN_SIZE), name='Wci')
    Wzi = SharedWeight(size=(2 * HIDDEN_SIZE, 2 * HIDDEN_SIZE), name='Wzi')
    Wcz = SharedWeight(size=(2 * HIDDEN_SIZE, 2 * HIDDEN_SIZE), name='Wcz')
    v = SharedWeight(size=(2 * HIDDEN_SIZE, 1), name='Wv')
    Wfu = SharedWeight(size=(4 * HIDDEN_SIZE, 2 * HIDDEN_SIZE), name='Wfu')

    Wbu = SharedWeight(size=(2 * HIDDEN_SIZE,), name='Wbu')

    Wgu = SharedWeight(size=(4 * HIDDEN_SIZE, 2 * HIDDEN_SIZE), name='Wgu')

    Wbg = SharedWeight(size=(2 * HIDDEN_SIZE,), name='Wbg')

    return Wci, Wzi, Wcz, v, Wfu, Wbu, Wgu, Wbg


def MneReader_Model(vocab_size, char_vocab_size, embedding_matrix, tag_size, ner_size, use_highway=True, share_sm=True):
    question_input = Input(shape=(None,), dtype='int32', name='question_input')
    context_input = Input(shape=(None,), dtype='int32', name='context_input')
    question_char = Input(shape=(None, max_char_size,),
                          dtype='int32', name='question_input_char')
    context_char = Input(shape=(None, max_char_size,),
                         dtype='int32', name='context_input_char')

    question_tag = Input(shape=(None,), dtype='int32',
                         name='question_tag_input')
    context_tag = Input(shape=(None,), dtype='int32', name='context_tag_input')

    question_ent = Input(shape=(None,), dtype='int32',
                         name='question_ent_input')
    context_ent = Input(shape=(None,), dtype='int32', name='context_ent_input')

    query_type_input = Input(shape=(1,), dtype='int32',
                             name='question_type_input')
    
    question_em = Input(shape=(None, 1), dtype='float32', name="question_em")
    context_em = Input(shape=(None, 1), dtype='float32', name="context_em")
    # contextual Embedding
    RNN = recurrent.LSTM
    if RNN_Cell == 'LSTM':
        RNN = recurrent.LSTM
    elif RNN_Cell == 'GRU':
        RNN = recurrent.GRU

    EMBEDDING_DIM = embedding_matrix.shape[-1]
    word_emb_layer = Embedding(output_dim=EMBEDDING_DIM, input_dim=vocab_size, weights=[embedding_matrix],
                               trainable=False, mask_zero=False)
    tag_emb_layer = Embedding(
        output_dim=tag_size, input_dim=tag_size, trainable=True, mask_zero=False)
    ner_emb_layer = Embedding(
        output_dim=ner_size, input_dim=ner_size, trainable=True, mask_zero=False)
    query_type_emb_layer = Embedding(
        output_dim=HIDDEN_SIZE // 2, input_dim=9, trainable=True, mask_zero=False)

    char_embedding_layer = TimeDistributed(Sequential([
        InputLayer(input_shape=(max_char_size,), dtype='int32'),
        Embedding(input_dim=char_vocab_size,
                  output_dim=CHAR_EMBEDDING_DIM, mask_zero=True),
        Bidirectional(RNN(units=CHAR_EMBEDDING_DIM, dropout=DP)),
    ]))  # 100

    def Encoder(word_input, tag_input, ent_input, char_input, type='query'):
        x1 = word_emb_layer(word_input)
        x2 = tag_emb_layer(tag_input)
        x3 = ner_emb_layer(ent_input)
        x4 = char_embedding_layer(char_input)
        x = concatenate([x1, x2, x3, x4])
        if type == 'query':
            x = concatenate([x, question_em])
            x5 = query_type_emb_layer(query_type_input)  # [B,1,H/2]
            x5 = Dense(EMBEDDING_DIM + tag_size + ner_size + HIDDEN_SIZE + 1)(x5)
            x = add([x, x5])
        else:
            x = concatenate([x, context_em])
        x = Dropout(DP)(x)
        return x

    u_question = Encoder(question_input, question_tag,
                         question_ent, question_char, type='query')
    u_context = Encoder(context_input, context_tag,
                        context_ent, context_char, type='context')

    context, question = u_context, u_question
    if share_sm:
        rnns = []
        for i in range(ENCODER_LAYERS):
            rnns.append(Bidirectional(
                RNN(HIDDEN_SIZE, return_sequences=True, dropout=DP)))
        for rnn in rnns:
            context = rnn(context)
            question = rnn(question)
    else:
        # context = Masking()(context)
        for i in range(ENCODER_LAYERS):
            context = Bidirectional(
                RNN(HIDDEN_SIZE, return_sequences=True, dropout=DP))(context)
            question = Bidirectional(RNN(HIDDEN_SIZE,
                                        return_sequences=True,
                                        dropout=DP))(question)
    context = Dropout(rate=DP, name='uP')(context)
    question = Dropout(rate=DP, name='uQ')(question)

    Cj = context
    Qi = question
    # Interative Aligner
    for t in range(HOPS):
        Cj = Interactive_Align_attention(
            name='Interactive_aligning_' + str(t))([Cj, Qi])
        Cj = Self_Align_attention(name='Self_aligning_' + str(t))(Cj)
        Cj = Bidirectional(RNN(HIDDEN_SIZE,
                               return_sequences=True,
                               dropout=DP))(Cj)  # [B,P,2d]
        Cj = Dropout(rate=DP)(Cj)  # model self-aware context represention

    # memory pointer
    Wci, Wzi, Wcz, v, Wfu, Wbu, Wgu, Wbg = get_share_weight_()
    shared_weights = [Wci, Wzi, Wcz, v, Wfu, Wbu, Wgu, Wbg]

    fake_input = GlobalMaxPooling1D()(
        Dense(2 * HIDDEN_SIZE, trainable=False)(u_context))  # not support mask
    fake_input = RepeatVector(n=2, name='fake_input')(fake_input)  # [B,2,2*H]

    Q_last = Bidirectional(RNN(HIDDEN_SIZE,
                               return_sequences=False,
                               dropout=DP))(Qi)
    Q_last = Dropout(rate=DP)(Q_last)
    zs = Q_last

    for t in range(HOPS):
        if t == HOPS - 1:
            ps = MemoryBasedPointer(units=2 * HIDDEN_SIZE,
                                    return_sequences=True,
                                    initial_state_provided=True,
                                    name='ps_last',
                                    unroll=UNROLL, is_last=True)([fake_input, Cj, Wci, Wzi, Wcz, v, Wfu, Wbu, Wgu, Wbg, zs])
        else:
            zs = MemoryBasedPointer(units=2 * HIDDEN_SIZE,
                                    return_sequences=False,
                                    initial_state_provided=True,
                                    name='zs_' + str(t),
                                    unroll=UNROLL, is_last=False)([fake_input, Cj, Wci, Wzi, Wcz, v, Wfu, Wbu, Wgu, Wbg, zs])

    answer_start = Slice(0, name='answer_start')(ps)
    answer_end = Slice(1, name='answer_end')(ps)  # [B,P]

    inputs = [question_input, question_char, question_tag, question_ent, question_em, query_type_input, context_input,
              context_char, context_tag, context_ent, context_em] + shared_weights
    outputs = [answer_start, answer_end]

    model = Model(input=inputs, output=outputs)
    return model
