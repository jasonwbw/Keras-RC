from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, InputLayer
from keras.layers.core import Dense, RepeatVector, Masking, Dropout, Lambda
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers import recurrent, Dense, Input, Dropout, TimeDistributed, Flatten, concatenate, Embedding, Merge, merge, multiply, add
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D

from utils.layers.core_layer import highway_network_wrapper, Linear_layer, softmax_layer, tile_layer, MaskedDense
from utils.layers.attention_layer4BiDAF import FasterBidirectionAttention, softselAttention, FasterSelf_Attention4BiDAF
from utils.layers.conv_layer import build_convs, multi_conv_withlayers


max_char_size = 16
# EMBEDDING_DIM = 100
CHAR_EMBEDDING_DIM = 75
CHAR_EMBED_METHOD = 'CNN'  # CNN
HIDDEN_SIZE = 100
MAX_SEQ_LEN = 400
ENCODER_LAYERS = 1
RNN_Cell = 'GRU'
DP = 0.2
UNROLL = False
MLP_LAYER = 1
ATT_TYPE = 'tri'


def BIDAF_Model(vocab_size, char_vocab_size, embedding_matrix, user_char_embed=True,
                use_highway=True, use_sefatt=False, share_sm=False):
    question_input = Input(shape=(None,), dtype='int32', name='question_input')
    context_input = Input(shape=(None,), dtype='int32', name='context_input')
    question_char = Input(shape=(None, max_char_size,),
                          dtype='int32', name='question_input_char')
    context_char = Input(shape=(None, max_char_size,),
                         dtype='int32', name='context_input_char')

    EMBEDDING_DIM = embedding_matrix.shape[-1]
    layer_emb = Embedding(output_dim=EMBEDDING_DIM, input_dim=vocab_size, weights=[embedding_matrix],
                          trainable=False, mask_zero=True)
    questionEmbd = layer_emb(question_input)
    contextEmbd = layer_emb(context_input)  # mask_zero=True,

    if user_char_embed:
        embedding_char = Embedding(
            output_dim=CHAR_EMBEDDING_DIM, input_dim=char_vocab_size, trainable=True, name='char_emb')  # no mask
        if CHAR_EMBED_METHOD == 'RNN':
            char_embedding_layer = TimeDistributed(Sequential([
                InputLayer(input_shape=(max_char_size,), dtype='int32'),
                embedding_char,
                Bidirectional(GRU(units=HIDDEN_SIZE))
            ]))  # 200
            context_charEmbed = char_embedding_layer(context_char)
            question_charEmbed = char_embedding_layer(question_char)
        else:
            context_charEmbed = embedding_char(context_char)
            question_charEmbed = embedding_char(question_char)
            convs, NGRAM_FILTERS = build_convs(NGRAM_FILTERS=[1,2,3,4], NUM_FILTER=HIDDEN_SIZE, name='char_cnn')
            context_charEmbed = multi_conv_withlayers(context_charEmbed, convs, NGRAM_FILTERS, name='char_context')  # [batch,len,4*H]
            question_charEmbed = multi_conv_withlayers(question_charEmbed, convs, NGRAM_FILTERS, name='char_query')  # out[batch,len,4*H]
        context = concatenate([context_charEmbed, contextEmbd])
        question = concatenate([question_charEmbed, questionEmbd])

    if use_highway:
        # the question and context dimension same
        cur_dim = context.get_shape().as_list()[-1]
        hn = highway_network_wrapper(cur_dim, 3)
        context = hn.build(context)
        question = hn.build(question)  # reuse

    # contextual Embedding
    RNN = recurrent.LSTM
    if RNN_Cell == 'LSTM':
        RNN = recurrent.LSTM
    elif RNN_Cell == 'GRU':
        RNN = recurrent.GRU

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

    BiAttenlayer = FasterBidirectionAttention(linear_fun=ATT_TYPE, is_q2c_att=True)

    P0 = BiAttenlayer([context, question])

    if use_sefatt:
        # P0 = Dropout(rate=DP, name='P0_drop')(P0)
        lP0 = MaskedDense(2 * HIDDEN_SIZE, activation='relu')(P0)
        pre_selfatt = Bidirectional(RNN(HIDDEN_SIZE, return_sequences=True, dropout=DP))(lP0)
        pre_selfatt = Dropout(rate=DP, name='pre_selfatt_drop')(pre_selfatt)
        after_selfatt = FasterSelf_Attention4BiDAF(linear_fun=ATT_TYPE)(pre_selfatt)
        # after_selfatt = Dropout(rate=DP, name='after_selfatt_drop')(after_selfatt)
        lafter_selfatt = MaskedDense(2 * HIDDEN_SIZE, activation='relu')(after_selfatt)
        g0 = add([lP0, lafter_selfatt])
    else:
        g0 = Bidirectional(RNN(HIDDEN_SIZE, return_sequences=True, dropout=DP))(P0)
    
    g1 = Bidirectional(RNN(HIDDEN_SIZE, return_sequences=True, dropout=DP))(g0)

    if use_sefatt:
        D1 = concatenate([g1, g0])
    else:
        D1 = concatenate([g1, P0])
    D1 = Dropout(DP)(D1)
    logits = Linear_layer(1, use_squeeze=True)(D1)  # start logits

    # for end inference
    ali = softselAttention()([g1, logits])  # [B,d]
    ali = tile_layer(1, g1)(ali)

    if use_sefatt:
        g2 = Bidirectional(RNN(HIDDEN_SIZE, return_sequences=True, dropout=DP))(concatenate([g0, g1, ali, multiply([g1, ali])]))
    else:
        g2 = Bidirectional(RNN(HIDDEN_SIZE, return_sequences=True, dropout=DP))(
                concatenate([P0, g1, ali, multiply([g1, ali])]))  # merge

    if use_sefatt:
        D2 = concatenate([g2, g0])
    else:
        D2 = concatenate([g2, P0])
    D2 = Dropout(DP)(D2)
    logits2 = Linear_layer(1, use_squeeze=True)(D2)
    
    yp = softmax_layer(axis=-1, name='answer_start')(logits)
    yp2 = softmax_layer(axis=-1, name='answer_end')(logits2)

    inputs = [question_input, question_char, context_input, context_char]
    outputs = [yp, yp2]
    model = Model(input=inputs,
                  output=outputs)
    return model
