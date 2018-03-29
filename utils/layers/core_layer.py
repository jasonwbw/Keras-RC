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
from keras.layers import Reshape,Permute,average,Dense,Flatten,multiply,add
from keras.models import Model


class MaskedDense(Dense):

    def compute_mask(self, input, mask=None):
        return mask


class MeanPooling(Layer):

    def __init__(self, axis, **kwargs):
        super(MeanPooling, self).__init__(**kwargs)
        self.axis = axis

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        return K.sum(x, axis=self.axis, keepdims=False) / (K.sum(K.cast(mask, 'float32'), axis=self.axis, keepdims=True) + K.epsilon())

    def compute_mask(self, input, mask=None):
        return None


class Linear_layer(Layer):
    def __init__(self,units,use_bias=False,use_squeeze=False,kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None,kernel_constraint=None,
                 bias_constraint=None,**kwargs):
        super(Linear_layer, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.units = units
        self.use_bias = use_bias
        self.use_squeeze = use_squeeze

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        pass

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        if self.use_squeeze:
            return tuple(output_shape[:-1])
        return tuple(output_shape)

    def call(self, x, mask=None):
        self.len = K.shape(x)[1]
        inputs = K.reshape(x,(-1,self.input_dim))
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        output = K.reshape(output,(-1,self.len,self.units))
        if self.use_squeeze:
            output = K.squeeze(output,axis=-1)
        return output

    def compute_mask(self, input, mask=None):
        return None



class softmax_layer(Layer):
    def __init__(self,axis, **kwargs):
        super(softmax_layer, self).__init__(**kwargs)
        self.axis = axis
        pass

    def build(self, input_shape):
        pass

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        if mask is None:
            mask = K.constant(True)
        mask = K.cast(mask, K.floatx())
        if K.ndim(x) is K.ndim(mask) + 1:
            mask = K.expand_dims(mask)

        m = K.max(x, axis=self.axis, keepdims=True)
        e = K.exp(x - m) * mask
        s = K.sum(e, axis=self.axis, keepdims=True)
        s += K.cast(K.cast(s < K.epsilon(), K.floatx()) * K.epsilon(), K.floatx())
        return e / s


class tile_layer(Layer):
    def __init__(self,axis,_tensor,**kwargs):
        super(tile_layer, self).__init__(**kwargs)
        self.axis = axis
        self._tensor = _tensor
        pass

    def build(self, input_shape):
        pass

    def compute_output_shape(self, input_shape):
        return tuple(self._tensor.get_shape().as_list())

    def call(self, x, mask=None):
        n = K.shape(self._tensor)[1]
        return  K.tile(K.expand_dims(x, 1), [1,n, 1])



class Gate_update_layer(Layer):
    def __init__(self,**kwargs):
        super(Gate_update_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, x, mask=None):
        gate = x[0]
        trans = x[1]
        orig = x[2]
        out = gate*trans + (1-gate)*orig
        return out


def linear(_in, output_size,name,DP=0):
    if DP>0:
        _in = Dropout(DP)(_in)
    out = Dense(output_size,use_bias=True,name=name)(_in)
    return out


def highway_layer(_in,name,DP=0):
    d = _in.get_shape().as_list()[-1]
    if DP>0:
        _in = Dropout(DP)(_in)
    trans_dense = Dense(d,use_bias=True,name=name+'_trans',activation='relu')
    gate_dense = Dense(d,use_bias=True,name=name+'_gate',activation='sigmoid')
    trans = trans_dense(_in)
    gate = gate_dense(_in)
    out = Gate_update_layer()([gate,trans,_in])
    return out


class highway_network_wrapper():
    def __init__(self,d,num_layers,name='highway'):
        self.d = d
        self.num_layers = num_layers
        self.trans_dense = []
        self.gate_dense = []
        self.gate_update = []
        for i in range(num_layers):
            self.trans_dense.append(Dense(d,use_bias=True,name=name+str(i)+'_trans',activation='relu'))
            self.gate_dense.append(Dense(d,use_bias=True,name=name+str(i)+'_gate',activation='sigmoid'))
            self.gate_update.append(Gate_update_layer(name=name+str(i)+'_update'))

    def highway_step(self,x,layer_index):
        trans = self.trans_dense[layer_index](x)
        gate = self.gate_dense[layer_index](x)
        out = self.gate_update[layer_index]([gate, trans, x])
        return out
    def build(self,arg):
        prev = arg
        cur = None
        for layer_idx in range(self.num_layers):
            cur = self.highway_step(prev,layer_idx)
            prev = cur
        return cur

if __name__ == '__main__':
    ss = Input((50, 128))
    yy = Input((50,128))
    with K.name_scope('highway'):
        hn = highway_network_wrapper(ss.get_shape().as_list()[-1],3)
        out_1 = hn.build(ss)
        out_2 = hn.build(yy)






#use for some layers which share same weights

from keras import backend as K

from keras import initializers
from keras import regularizers

from keras.engine.topology import Node
from keras.layers import Layer, InputLayer

class SharedWeightLayer(InputLayer):
    def __init__(self,
                 size,
                 initializer='glorot_uniform',
                 regularizer=None,
                 name=None,
                 **kwargs):
        self.size = tuple(size)
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)

        if not name:
            prefix = 'shared_weight'
            name = prefix + '_' + str(K.get_uid(prefix))

        Layer.__init__(self, name=name, **kwargs)

        with K.name_scope(self.name):#self add weight define
            self.kernel = self.add_weight(shape=self.size,
                                          initializer=self.initializer,
                                          name='kernel',
                                          regularizer=self.regularizer)

        self.trainable = True
        self.built = True
        # self.sparse = sparse

        input_tensor = self.kernel * 1.0

        self.is_placeholder = False
        input_tensor._keras_shape = self.size

        input_tensor._uses_learning_phase = False
        input_tensor._keras_history = (self, 0, 0)

        Node(self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=[input_tensor],
             output_tensors=[input_tensor],
             input_masks=[None],
             output_masks=[None],
             input_shapes=[self.size],
             output_shapes=[self.size])

    def get_config(self):
        config = {
            'size': self.size,
            'initializer': initializers.serialize(self.initializer),
            'regularizer': regularizers.serialize(self.regularizer)
        }
        base_config = Layer.get_config(self)
        return dict(list(base_config.items()) + list(config.items()))


def SharedWeight(**kwargs):
    input_layer = SharedWeightLayer(**kwargs)

    outputs = input_layer.inbound_nodes[0].output_tensors
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


class Slice(Layer):
    def __init__(self, indices, axis=1, **kwargs):
        self.supports_masking = True
        self.axis = axis

        if isinstance(indices, slice):
            self.indices = (indices.start, indices.stop, indices.step)
        else:
            self.indices = indices

        self.slices = [slice(None)] * self.axis#axis is can be sliced
        #befor axis is keep same dimension

        if isinstance(self.indices, int):
            self.slices.append(self.indices)
        elif isinstance(self.indices, (list, tuple)):
            self.slices.append(slice(*self.indices))
        else:
            raise TypeError("indices must be int or slice")

        super(Slice, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        return inputs[self.slices]

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        for i, slice in enumerate(self.slices):
            if i == self.axis:
                continue
            start = slice.start or 0
            stop = slice.stop or input_shape[i]
            step = slice.step or 1
            input_shape[i] = None if stop is None else (stop - start) // step
        del input_shape[self.axis]

        return tuple(input_shape)

    def compute_mask(self, x, mask=None):
        if mask is None:
            return mask
        if self.axis == 1:
            return mask[self.slices]
        else:
            return mask

    def get_config(self):
        config = {'axis': self.axis,
                  'indices': self.indices}
        base_config = super(Slice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




