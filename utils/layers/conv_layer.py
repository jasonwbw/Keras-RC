
import numpy as np

from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import *
# from visualizer import *
from keras.models import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical#, accuracy
from keras.layers.core import *
from keras.layers import Conv1D,Conv2D,MaxPooling1D
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D,MaxPool2D,GlobalMaxPool2D
from keras.layers import concatenate
from keras.layers import Reshape,Permute,average


class Global_one_dim_pooling(Layer):
    def __init__(self, dim_num=2,method='max',**kwargs):
        super(Global_one_dim_pooling, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)
        self.dim_num = dim_num
        self.method = method

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1],input_shape[3])

    def call(self, inputs):
        if self.method=='max':
            return K.max(inputs, axis=[self.dim_num])
        elif self.method=='mean':
            return K.mean(inputs,axis=[self.dim_num])


def build_convs(NGRAM_FILTERS=[1,2,3,4], NUM_FILTER=128, padding='valid', act='relu', name='char'):
    convolutions = []
    i = 0
    for n_gram in NGRAM_FILTERS:
        i += 1
        conv_in = Conv2D(filters=NUM_FILTER,kernel_size=(1,n_gram),padding=padding,
            activation=act,strides=(1,1),name="conv_" +name+ str(n_gram) + '_' + str(i))
        convolutions.append(conv_in)
    return convolutions, NGRAM_FILTERS


def multi_conv_withlayers(_in, conv_layers, NGRAM_FILTERS, pooling='max', name='char', DP=0.1):
    convolutions = []
    i = 0
    for n_gram in NGRAM_FILTERS:
        conv_in = conv_layers[i](_in)
        # pool
        if pooling=='max':
            _pooling = Global_one_dim_pooling(dim_num=2,method='max')(conv_in)
        elif pooling=='mean':
            _pooling = Global_one_dim_pooling(dim_num=2,method='mean')(conv_in)
        i += 1
        # flattened = Flatten()(one_max)
        convolutions.append(_pooling)
    out = concatenate(convolutions, name='match_concat_'+name)
    out = Dropout(DP)(out)
    return out


def multi_conv(_in,NGRAM_FILTERS=[1,2,3,4],NUM_FILTER=128,pooling='max',padding='valid',act='relu',DP=0.1,name='char'):
    convolutions = []
    i = 0
    for n_gram in NGRAM_FILTERS:
        i += 1
        conv_in = Conv2D(filters=NUM_FILTER,kernel_size=(1,n_gram),padding=padding,
            activation=act,strides=(1,1),name="conv_" +name+ str(n_gram) + '_' + str(i))(_in)
        # pool
        if pooling=='max':
            _pooling = Global_one_dim_pooling(dim_num=2,method='max')(conv_in)
        elif pooling=='mean':
            _pooling = Global_one_dim_pooling(dim_num=2,method='mean')(conv_in)
        # flattened = Flatten()(one_max)
        convolutions.append(_pooling)
    out = concatenate(convolutions, name='match_concat_'+name)
    out = Dropout(DP)(out)
    return out
