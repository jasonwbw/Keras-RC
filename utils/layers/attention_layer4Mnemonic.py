
import numpy as np

from keras import backend as K
from keras.layers.recurrent import GRU
from keras.layers import concatenate, multiply, Layer

from keras import initializers, activations

import tensorflow as tf

from .attention_utils import softmax, exp_mask, compute_mask


def _sfu(r, fs, Wr, Br, Wg, Bg):
    c_ = concatenate([r] + fs)  # [B,JX,4d]
    r_ = K.tanh(K.dot(c_, Wr) + Br) #[B,Jx,d]
    g = K.sigmoid(K.dot(c_, Wg) + Bg)
    out = g * r_ + (1-g) * r
    return out


def _get_logits(h_aug, u_aug, selfatt=False):
    """
    dot function
    :param h_aug: [B,JX,2d]
    :param u_aug: [B,JQ,2d]
    :return:
    """
    out = K.batch_dot(h_aug, K.permute_dimensions(u_aug, (0, 2, 1)))   # [batch_size,JX,JQ]
    if selfatt:
        return tf.matrix_set_diag(out, tf.zeros_like(out[:,:,0]))
    return out


def _softsel(targets, logits):
    """
    :param targets: [B,JQ,2d]
    :param logits:  [B,JX,JQ]
    :return: [B,JQ,2d]
    """
    a = softmax(logits, axis=-1)
    e_i = K.batch_dot(a, targets)   # [batch_size,JX,d]
    return e_i


#attention layer for Mnemonic reader
class Interactive_Align_attention(Layer):#attention pooling layers use ones as previous
    def __init__(self,kernel_initializer='glorot_uniform',bias_initializer='zeros',**kwargs):
        """
        :param linear_fun: ['tri,dot,bil'] methods for computing the 2-dimention of query-context matrix
        :param kernel_initializer:
        :param kwargs:
        """
        super(Interactive_Align_attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def compute_output_shape(self, input_shape):
        assert (isinstance(input_shape, list) and len(input_shape) == 2)
        input_shape = input_shape[0]
        B, P, H = input_shape
        return (B,P,H)

    def build(self, input_shape):
        assert (isinstance(input_shape, list) and len(input_shape) == 2)
        H = input_shape[0]
        self.d = H[-1]

        self.Wr = self.add_weight((4 * self.d, self.d),
                                      initializer=self.kernel_initializer,
                                      name='{}_Wr'.format(self.name))

        self.Br = self.add_weight((self.d,),initializer=self.bias_initializer,
                                      name='{}_Br'.format(self.name))

        self.Wg = self.add_weight((4 * self.d, self.d),
                                      initializer=self.kernel_initializer,
                                      name='{}_Wg'.format(self.name))

        self.Bg = self.add_weight((self.d,),initializer=self.bias_initializer,
                                      name='{}_Bg'.format(self.name))
        self.built = True

    def call(self, inputs, mask=None): #question based on the parameters VrQ
        c = inputs[0]#context [batch,P,2d]
        q = inputs[1]#query [batch,Q,2d]

        self.JX = K.shape(c)[1]
        self.JQ = K.shape(q)[1]

        #tri linear
        B = _get_logits(c, q)# coattention matrix

        if mask is not None: #necessary for long length input
            c_mask = mask[0]
            q_mask = mask[1]
            # mask need expand
            c_mask_aug = K.tile(K.expand_dims(c_mask, axis=2), [1, 1, self.JQ])
            q_mask_aug = K.tile(K.expand_dims(q_mask, axis=1), [1, self.JX, 1])
            cq_mask = c_mask_aug & q_mask_aug  # mask是都
            #add mask
            B = exp_mask(B, cq_mask)

        q_a = _softsel(q, B)  # attened query vector [B,P,2d] for all context words
        #SFU
        out = _sfu(c, [q_a, c*q_a, c-q_a], self.Wr, self.Br, self.Wg, self.Bg)
        return out

    def compute_mask(self, input, mask=None):
        return mask[0]



# class Self_Align_attention(Layer):#attention pooling layers use ones as previous
#     def __init__(self,kernel_initializer='glorot_uniform',bias_initializer='zeros',**kwargs):
#         """
#         :param linear_fun: ['tri,dot,bil'] methods for computing the 2-dimention of query-context matrix
#         :param kernel_initializer:
#         :param kwargs:
#         """
#         super(Self_Align_attention, self).__init__(**kwargs)
#         self.supports_masking = True
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)

#     def compute_output_shape(self, input_shape):
#         assert (isinstance(input_shape, list) and len(input_shape) == 2)
#         input_shape = input_shape[0]
#         B, P, H = input_shape
#         return (B,P,H)

#     def build(self, input_shape):
#         assert (isinstance(input_shape, list) and len(input_shape) == 2)
#         H = input_shape[0]
#         self.d = H[-1]

#         self.Wr = self.add_weight((4 * self.d, self.d),
#                                       initializer=self.kernel_initializer,
#                                       name='{}_Wr'.format(self.name))

#         self.Br = self.add_weight((self.d,),initializer=self.bias_initializer,
#                                       name='{}_Br'.format(self.name))

#         self.Wg = self.add_weight((4 * self.d, self.d),
#                                       initializer=self.kernel_initializer,
#                                       name='{}_Wg'.format(self.name))

#         self.Bg = self.add_weight((self.d,),initializer=self.bias_initializer,
#                                       name='{}_Bg'.format(self.name))
#         self.built = True

#     def call(self, inputs, mask=None): #question based on the parameters VrQ
#         c = inputs[0]#context [batch,P,2d]
#         q = inputs[1]#query [batch,Q,2d]

#         self.JX = K.shape(c)[1]
#         self.JQ = K.shape(q)[1]

#         #tri linear
#         B = _get_logits(c, q, True)# coattention matrix

#         if mask is not None: #necessary for long length input
#             c_mask = mask[0]
#             q_mask = mask[1]
#             # mask need expand
#             c_mask_aug = K.tile(K.expand_dims(c_mask, axis=2), [1, 1, self.JQ])
#             q_mask_aug = K.tile(K.expand_dims(q_mask, axis=1), [1, self.JX, 1])
#             cq_mask = c_mask_aug & q_mask_aug  # mask
#             #add mask
#             B = exp_mask(B, cq_mask)

#         q_a = _softsel(q, B)  # attened query vector [B,P,2d] for all context words
#         #SFU
#         out = _sfu(c, [q_a, c*q_a, c-q_a], self.Wr, self.Br, self.Wg, self.Bg)
#         return out

#     def compute_mask(self, input, mask=None):
#         return mask[0]

class Self_Align_attention(Layer):#attention pooling layers use ones as previous
    def __init__(self,kernel_initializer='glorot_uniform',bias_initializer='zeros',**kwargs):
        """
        :param linear_fun: ['tri,dot,bil'] methods for computing the 2-dimention of query-context matrix
        :param kernel_initializer:
        :param kwargs:
        """
        super(Self_Align_attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def compute_output_shape(self, input_shape):
        B, P, H = input_shape
        return (B,P,H)

    def build(self, input_shape):
        H = input_shape
        self.d = H[-1]

        self.Wr = self.add_weight((4 * self.d, self.d),
                                      initializer=self.kernel_initializer,
                                      name='{}_Wr'.format(self.name))

        self.Br = self.add_weight((self.d,),initializer=self.bias_initializer,
                                      name='{}_Br'.format(self.name))

        self.Wg = self.add_weight((4 * self.d, self.d),
                                      initializer=self.kernel_initializer,
                                      name='{}_Wg'.format(self.name))

        self.Bg = self.add_weight((self.d,),initializer=self.bias_initializer,
                                      name='{}_Bg'.format(self.name))
        self.built = True

    def call(self, inputs, mask=None): #question based on the parameters VrQ
        c = inputs   #context [batch,P,2d]

        self.JX = K.shape(c)[1]

        #tri linear
        B = _get_logits(c, c, True)# coattention matrix

        if mask is not None: #necessary for long length input
            c_mask = mask
            # mask need expand
            c_mask_aug = K.tile(K.expand_dims(c_mask, axis=2), [1, 1, self.JX])
            q_mask_aug = K.tile(K.expand_dims(c_mask, axis=1), [1, self.JX, 1])
            cq_mask = c_mask_aug & q_mask_aug  # mask
            #add mask
            B = exp_mask(B, cq_mask)

        q_a = _softsel(c, B)  # attened query vector [B,P,2d] for all context words
        #SFU
        out = _sfu(c, [q_a, c*q_a, c-q_a], self.Wr, self.Br, self.Wg, self.Bg)
        return out

    def compute_mask(self, input, mask=None):
        return mask


class WrappedGRU(GRU):
    def __init__(self, initial_state_provided=False, **kwargs):
        kwargs['implementation'] = kwargs.get('implementation', 2)
        assert (kwargs['implementation'] == 2)

        super(WrappedGRU, self).__init__(**kwargs)
        self.input_spec = None
        self.initial_state_provided = initial_state_provided

    def call(self, inputs, mask=None, training=None, initial_state=None):
        if self.initial_state_provided:
            initial_state = inputs[-1:]#init_state set RQ
            inputs = inputs[:-1]

            # initial_state_mask = mask[-1:]
            mask = mask[:-1] if mask is not None else None

        self._non_sequences = inputs[1:]
        inputs = inputs[:1]#truncked for the current GRU cell

        self._mask_non_sequences = []
        if mask is not None:
            self._mask_non_sequences = mask[1:]
            mask = mask[:1]#only need question
        self._mask_non_sequences = [mask for mask in self._mask_non_sequences
                                    if mask is not None]

        if self.initial_state_provided:
            assert (len(inputs) == len(initial_state))
            inputs += initial_state

        if len(inputs) == 1:
            inputs = inputs[0]

        if isinstance(mask, list) and len(mask) == 1:
            mask = mask[0]

        return super(WrappedGRU, self).call(inputs, mask, training)

    def get_constants(self, inputs, training=None):#step_function(inp, states + constants) each step used
        constants = super(WrappedGRU, self).get_constants(inputs, training=training)
        constants += self._non_sequences
        constants += self._mask_non_sequences
        return constants

    def get_config(self):
        config = {'initial_state_provided': self.initial_state_provided}
        base_config = super(WrappedGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MemoryBasedPointer(WrappedGRU):
    def __init__(self,is_last=False, **kwargs):
        self.is_last=is_last
        super(MemoryBasedPointer, self).__init__(**kwargs)

    def build(self, input_shape):
        H = self.units // 2
        assert (isinstance(input_shape, list))

        nb_inputs = len(input_shape)
        assert (nb_inputs >= 5)

        assert (len(input_shape[0]) >= 2)
        B, T = input_shape[0][:2]

        assert (len(input_shape[1]) == 3)
        B, P, H_ = input_shape[1]
        assert (H_ == 2 * H)

        self.input_spec = [None]
        super(MemoryBasedPointer, self).build(input_shape=(B, T, 2 * H))
        self.input_spec = [None] * nb_inputs  # TODO TODO TODO

    def step(self, inputs, states):
        # input
        zs = states[0]  # (B, 2H) memory
        _ = states[1:3]  # ignore internal dropout/masks
        ci, Wci, Wzi, Wcz, v, Wfu, Wbu, Wgu, Wbg = states[3:12]
        ci_mask = states[12:13]

        zs_aug = K.tile(K.expand_dims(zs,axis=1),[1,K.shape(ci)[1],1])#[B,P,2H]
        cz = zs_aug*ci

        si = K.tanh(K.dot(ci,Wci) + K.dot(zs_aug,Wzi) + K.dot(cz,Wcz))#[B,P,2H]
        s_t = K.dot(si, v)  # (B, P, 1)
        s_t = K.batch_flatten(s_t)  # (B, P)
        a_t = K.softmax(s_t)#softmax(s_t,axis=1)#[B,P]
        us = K.batch_dot(ci,a_t,axes=[1,1])#[B,2H]

        #redefine
        ze = _sfu(zs, [us], Wfu, Wbu, Wgu, Wbg)#[B,2H]

        Net_inputs = ze
        ha_t, (ha_t_,)= super(MemoryBasedPointer, self).step(Net_inputs, states)#[B,2H]

        if self.is_last:#is last hop
            return a_t, [ha_t]

        return ha_t, [ha_t]#each timestep output is a_t (B, P) each

    def compute_output_shape(self, input_shape):
        assert (isinstance(input_shape, list))

        nb_inputs = len(input_shape)
        assert (nb_inputs >= 5)

        assert (len(input_shape[0]) >= 2)
        B, T = input_shape[0][:2]

        assert (len(input_shape[1]) == 3)
        B, P, H_ = input_shape[1]

        if self.is_last:
            if self.return_sequences:
                return (B, T, P)
            else:
                return (B, P)
        else:
            return (B, H_)

    def compute_mask(self, inputs, mask=None):
        return mask[3]
