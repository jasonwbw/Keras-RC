
import numpy as np

from keras import backend as K
from keras.layers.recurrent import GRU
from keras.layers import concatenate, multiply, Layer

from keras import initializers, activations

import tensorflow as tf

# attention for rnet
def softmax(x, axis, mask=None):
    if mask is None:
        mask = K.constant(True)
    mask = K.cast(mask, K.floatx())
    if K.ndim(x) is K.ndim(mask) + 1:
        mask = K.expand_dims(mask)

    m = K.max(x, axis=axis, keepdims=True)
    e = K.exp(x - m) * mask
    s = K.sum(e, axis=axis, keepdims=True)
    s += K.cast(K.cast(s < K.epsilon(), K.floatx()) * K.epsilon(), K.floatx())
    return e / s


def compute_mask(x, mask_value=0):
    boolean_mask = K.any(K.not_equal(x, mask_value), axis=-1, keepdims=False)
    return K.cast(boolean_mask, K.floatx())


VERY_BIG_NUMBER = 1e30
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


def exp_mask(val, mask, name=None):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor

    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    if name is None:
        name = "exp_mask"
    import tensorflow as tf
    return tf.add(val, (1 - K.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name=name)


class softselAttention(Layer):
    """
    use logits to attend the context and reduce information
    """

    def __init__(self, **kwargs):
        super(softselAttention, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        assert (isinstance(input_shape, list) and len(input_shape) == 2)
        input_shape = list(input_shape[0])
        return tuple(input_shape[:-2] + [input_shape[-1]])

    def build(self, input_shape):
        assert (isinstance(input_shape, list) and len(input_shape) == 2)
        H = input_shape[0]
        U = input_shape[1]
        self.JQ = H[1]
        self.d = H[-1]
        self.built = True

    def call(self, inputs, mask=None):  # question based on the parameters VrQ
        """
        :param targets: [ B,JX,JQ,2d]
        :param logits:  [B,JX,JQ]
        :return: [B,JX,2d]
        """
        targets = inputs[0]
        logits = inputs[1]

        a = softmax(logits, axis=-1)
        e_t = K.expand_dims(a, axis=-1)  # [batch_size,JX,JQ,1]
        e_i = multiply([e_t, targets])  # [batch_size,JX,JQ,d]
        return K.sum(e_i, axis=-2)

    def compute_mask(self, input, mask=None):
        return None


# attention pooling layers use ones as previous
class BidirectionAttention(Layer):
    def __init__(self, linear_fun='tri', kernel_initializer='glorot_uniform', is_q2c_att=True, **kwargs):
        """
        :param linear_fun: ['tri,dot,bil'] methods for computing the 2-dimention of query-context matrix
        :param kernel_initializer:
        :param is_q2c_att: if return bilteral flow
        :param kwargs:
        """
        super(BidirectionAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.linear_fun = linear_fun
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.is_q2c_att = is_q2c_att
        self.att_active = activations.get('relu')

    def compute_output_shape(self, input_shape):
        assert (isinstance(input_shape, list) and len(input_shape) == 2)
        input_shape = input_shape[0]
        B, P, H = input_shape

        if self.is_q2c_att:
            return (B, P, 4 * H)
        else:
            return (B, P, 3 * H)

    def build(self, input_shape):
        assert (isinstance(input_shape, list) and len(input_shape) == 2)
        H = input_shape[0]
        self.d = H[-1]

        if self.linear_fun == 'tri':
            self.Wa = self.add_weight((3 * self.d, 1),
                                      initializer=self.kernel_initializer,
                                      name='{}_Wa'.format(self.name))
        elif self.linear_fun != 'dot':
            self.Wa = self.add_weight((2 * self.d, 1),
                                      initializer=self.kernel_initializer,
                                      name='{}_Wa'.format(self.name))
        self.built = True

    def _softsel(self, targets, logits, is4u=True):
        """
        :param targets: [ B,JX,JQ,2d]
        :param logits:  [B,JX,JQ]
        :return: [B,JX,2d]
        """
        a = softmax(logits, axis=-1)
        if not is4u:
            e_t = K.expand_dims(a, axis=-1)  # [batch_size,JX,JQ,1]
            e_i = multiply([e_t, targets])  # [batch_size,JX,JQ,d]
            return K.sum(e_i, axis=-2)
        else:
            return K.batch_dot(a, targets)

    def _get_logits(self, h_aug, u_aug):
        """
        :param h_aug: [ ...,2d]
        :param u_aug: [....,2d]
        :return:
        """
        if self.linear_fun == 'tri':
            new_aug = h_aug * u_aug  # [",2d]
            aug = concatenate([K.reshape(h_aug, [-1, self.d]), K.reshape(u_aug,
                                                                         [-1, self.d]), K.reshape(new_aug, [-1, self.d])])  # [?,6d]
            out = K.dot(aug, self.Wa)
            out = K.reshape(out, [-1, self.JX, self.JQ])
            return out
        elif self.linear_fun == 'dot':
            new_aug = h_aug * u_aug
            rank = len(new_aug.get_shape())
            out = K.sum(new_aug, rank - 1)
            return out
        else:  # double linear
            aug = concatenate([K.reshape(h_aug, [-1, self.d]),
                               K.reshape(u_aug, [-1, self.d])])  # [?,4d]
            out = K.dot(aug, self.Wa)
            out = K.reshape(out, [-1, self.JX, self.JQ])
            return out

    def call(self, inputs, mask=None):  # question based on the parameters VrQ
        h = inputs[0]  # context [batch,P,2d]
        u = inputs[1]  # query [batch,Q,2d]

        self.JX = K.shape(h)[1]
        self.JQ = K.shape(u)[1]

        h_aug = K.tile(K.expand_dims(h, axis=2), [1, 1, self.JQ, 1])
        u_aug = K.tile(K.expand_dims(u, axis=1), [1, self.JX, 1, 1])
        # tri linear
        u_logits = self._get_logits(h_aug, u_aug)  # how to combine

        if mask is not None:  # necessary for long length input
            h_mask = mask[0]
            u_mask = mask[1]
            # mask need expand
            h_mask_aug = K.tile(K.expand_dims(h_mask, axis=2), [1, 1, self.JQ])
            u_mask_aug = K.tile(K.expand_dims(u_mask, axis=1), [1, self.JX, 1])
            hu_mask = h_mask_aug & u_mask_aug  # mask是都
            # add mask
            u_logits = exp_mask(u_logits, hu_mask)

        u_a = self._softsel(u_aug, u_logits)  # [N,JX,2d] 2d*T
        h_a = self._softsel(h, K.max(u_logits, axis=2), is4u=False)  # [N,2d]
        h_a = K.tile(K.expand_dims(h_a, 1), [1, self.JX, 1])  # [N,JX,2d] 2d*T

        if self.is_q2c_att:
            p0 = concatenate([h, u_a, h * u_a, h * h_a])  # 8d*T
        else:
            p0 = concatenate([h, u_a, h * u_a])  # 6d*T

        return p0

    def compute_mask(self, input, mask=None):
        if mask is not None:
            return mask[0]
        return None


class FasterBidirectionAttention(BidirectionAttention):

    def build(self, input_shape):
        assert (isinstance(input_shape, list) and len(input_shape) == 2)
        H = input_shape[0]
        self.d = H[-1]

        if self.linear_fun == 'tri':
            self.Wa_1 = self.add_weight((self.d, 1),
                                      initializer=self.kernel_initializer,
                                      name='{}_Wa_1'.format(self.name))
            self.Wa_2 = self.add_weight((self.d, 1),
                                      initializer=self.kernel_initializer,
                                      name='{}_Wa_2'.format(self.name))
            self.Wa_3 = self.add_weight((1, self.d),
                                      initializer=self.kernel_initializer,
                                      name='{}_Wa_3'.format(self.name))
            self.Wa_3_for_compute = K.expand_dims(self.Wa_3, axis=0)
            self.ones = self.add_weight((1, 767),
                                      initializer=initializers.get('ones'),
                                      name='{}_ones'.format(self.name))
        else:
            super(FasterBidirectionAttention, self).build(input_shape)
        self.built = True

    def _get_logits(self, h_aug, u_aug):
        """
        :param h_aug: [ ...,2d]
        :param u_aug: [....,2d]
        :return:
        """
        if self.linear_fun == 'tri':
            ones_h = self.ones[:, :self.JQ]
            trans_h = K.dot(K.dot(h_aug, self.Wa_1), ones_h)   # [batch_size,JX,JQ]
            ones_u = self.ones[:, :self.JX]
            trans_u = K.permute_dimensions(K.dot(K.dot(u_aug, self.Wa_2), ones_u), (0, 2, 1))   # [batch_size,JX,JQ]
            trans_hu = K.batch_dot(h_aug * self.Wa_3_for_compute, K.permute_dimensions(u_aug, (0, 2, 1)))   # [batch_size,JX,JQ]
            out = trans_h + trans_u + trans_hu
            return out
        else:
            return super(FasterBidirectionAttention, self)._get_logits(h_aug, u_aug)
    
    def _softsel(self, targets, logits, is4u=True):
        """
        :param targets: [ B,JX,JQ,2d]
        :param logits:  [B,JX,JQ]
        :return: [B,JX,2d]
        """
        a = softmax(logits, axis=-1)
        if self.linear_fun != 'tri' or not is4u:
            e_t = K.expand_dims(a, axis=-1)  # [batch_size,JX,JQ,1]
            e_i = multiply([e_t, targets])  # [batch_size,JX,JQ,d]
            return K.sum(e_i, axis=-2)
        else:
            return K.batch_dot(a, targets)

    def call(self, inputs, mask=None):  # question based on the parameters VrQ
        h = inputs[0]  # context [batch,P,2d]
        u = inputs[1]  # query [batch,Q,2d]

        if self.linear_fun != 'tri':
            return super(FasterBidirectionAttention).call(inputs, mask=mask)

        self.JX = K.shape(h)[1]
        self.JQ = K.shape(u)[1]
        u_logits = self._get_logits(h, u)  # how to combine

        if mask is not None:  # necessary for long length input
            h_mask = mask[0]
            u_mask = mask[1]
            # mask need expand
            h_mask_aug = K.tile(K.expand_dims(h_mask, axis=2), [1, 1, self.JQ])
            u_mask_aug = K.tile(K.expand_dims(u_mask, axis=1), [1, self.JX, 1])
            hu_mask = h_mask_aug & u_mask_aug  # mask是都
            # add mask
            u_logits = exp_mask(u_logits, hu_mask)

        u_a = self._softsel(u, u_logits)
        h_a = self._softsel(h, K.max(u_logits, axis=2), is4u=False)  # [N,2d]
        h_a = K.tile(K.expand_dims(h_a, 1), [1, self.JX, 1])  # [N,JX,2d] 2d*T

        if self.is_q2c_att:
            p0 = concatenate([h, u_a, h * u_a, h * h_a])  # 8d*T
        else:
            p0 = concatenate([h, u_a, h * u_a])  # 6d*T

        return p0


# attention pooling layers use ones as previous
class Self_Attention4BiDAF(Layer):
    def __init__(self, linear_fun='dot', kernel_initializer='glorot_uniform', sub_d=50, **kwargs):
        """
        :param linear_fun: ['tri,dot,bil'] methods for computing the 2-dimention of query-context matrix
        :param kernel_initializer:
        :param is_q2c_att: if return bilteral flow
        :param kwargs:
        """
        super(Self_Attention4BiDAF, self).__init__(**kwargs)
        self.supports_masking = True
        self.linear_fun = linear_fun
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.sub_d = sub_d
        self.att_active = activations.get('relu')

    def compute_output_shape(self, input_shape):
        B, P, H = input_shape

        return (B, P, (3 if self.linear_fun != 'fusion' else 2) * H)

    def build(self, input_shape):
        H = input_shape
        self.d = H[-1]

        if self.linear_fun == 'tri':
            self.Wa = self.add_weight((3 * self.d, 1),
                                      initializer=self.kernel_initializer,
                                      name='{}_Wa'.format(self.name))

        self.built = True

    def _softsel(self, targets, logits):
        """
        :param targets: [ B,JX,JQ,2d]
        :param logits:  [B,JX,JQ]
        :return: [B,JX,2d]
        """
        a = softmax(logits, axis=-1)
        if self.linear_fun != 'fusion':
            e_t = K.expand_dims(a, axis=-1)  # [batch_size,JX,JQ,1]
            e_i = multiply([e_t, targets])  # [batch_size,JX,JQ,d]
            return K.sum(e_i, axis=-2)
        else:
            return K.batch_dot(a, targets)

    def _get_logits(self, h_aug, u_aug):
        """
        :param h_aug: [ ...,2d]
        :param u_aug: [....,2d]
        :return:
        """
        if self.linear_fun == 'tri':
            new_aug = h_aug * u_aug  # [",2d]
            aug = concatenate([K.reshape(h_aug, [-1, self.d]), K.reshape(u_aug,
                                                                         [-1, self.d]), K.reshape(new_aug, [-1, self.d])])  # [?,6d]
            out = K.dot(aug, self.Wa)
            out = K.reshape(out, [-1, self.JX, self.JX])
            return out
        elif self.linear_fun == 'dot':
            new_aug = h_aug * u_aug
            rank = len(new_aug.get_shape())
            out = K.sum(new_aug, rank - 1)
            return out

    def call(self, inputs, mask=None):  # question based on the parameters VrQ
        h = inputs  # context [batch,P,2d]

        self.JX = K.shape(h)[1]

        h_aug = K.tile(K.expand_dims(h, axis=2), [1, 1, self.JX, 1])
        u_aug = K.tile(K.expand_dims(h, axis=1), [1, self.JX, 1, 1])
        # tri linear
        u_logits = self._get_logits(h_aug, u_aug)  # how to combine
        
        if mask is not None:  # necessary for long length input
            h_mask = mask
            # mask need expand
            h_mask_aug = K.tile(K.expand_dims(h_mask, axis=2), [1, 1, self.JX])
            u_mask_aug = K.tile(K.expand_dims(h_mask, axis=1), [1, self.JX, 1])
            hu_mask = h_mask_aug & u_mask_aug  # mask是都
            # diagonal_mask
            diag_ones = K.ones_like(h_mask_aug, dtype='float32')
            diag_zeros = K.zeros_like(h_mask_aug, dtype='float32')
            diag = tf.matrix_diag_part(diag_ones)
            diag_mask = tf.matrix_set_diag(diag_zeros, diag)
            hud_mask = hu_mask & K.cast(diag_mask, 'bool')
            # add mask
            u_logits = exp_mask(u_logits, hud_mask)

        u_a = self._softsel(u_aug, u_logits)  # [N,JX,2d] 2d*T

        p0 = concatenate([h, u_a, h * u_a])  # 6d*T

        return p0

    def compute_mask(self, input, mask=None):
        return mask


class FasterSelf_Attention4BiDAF(Self_Attention4BiDAF):

    def build(self, input_shape):
        H = input_shape
        self.d = H[-1]

        if self.linear_fun == 'tri':
            self.Wa_1 = self.add_weight((self.d, 1),
                                      initializer=self.kernel_initializer,
                                      name='{}_Wa_1'.format(self.name))
            self.Wa_2 = self.add_weight((self.d, 1),
                                      initializer=self.kernel_initializer,
                                      name='{}_Wa_2'.format(self.name))
            self.Wa_3 = self.add_weight((1, self.d),
                                      initializer=self.kernel_initializer,
                                      name='{}_Wa_3'.format(self.name))
            self.Wa_3_for_compute = K.expand_dims(self.Wa_3, axis=0)
            self.ones = self.add_weight((1, 767),
                                      initializer=initializers.get('ones'),
                                      name='{}_ones'.format(self.name))
        else:
            super(FasterSelf_Attention4BiDAF, self).build(input_shape)
        self.built = True

    def _get_logits(self, h_aug, u_aug):
        """
        :param h_aug: [ ...,2d]
        :param u_aug: [....,2d]
        :return:
        """
        if self.linear_fun == 'tri':
            ones_h = self.ones[:, :self.JX]
            trans_h = K.dot(K.dot(h_aug, self.Wa_1), ones_h)   # [batch_size,JX,JQ]
            ones_u = self.ones[:, :self.JX]
            trans_u = K.permute_dimensions(K.dot(K.dot(u_aug, self.Wa_2), ones_u), (0, 2, 1))   # [batch_size,JX,JQ]
            trans_hu = K.batch_dot(h_aug * self.Wa_3_for_compute, K.permute_dimensions(u_aug, (0, 2, 1)))   # [batch_size,JX,JQ]
            out = trans_h + trans_u + trans_hu
            return out
        else:
            return super(FasterSelf_Attention4BiDAF, self)._get_logits(h_aug, u_aug)
    
    def _softsel(self, targets, logits):
        """
        :param targets: [ B,JX,JQ,2d]
        :param logits:  [B,JX,JQ]
        :return: [B,JX,2d]
        """
        a = softmax(logits, axis=-1)
        if self.linear_fun != 'fusion' and self.linear_fun != 'tri':
            e_t = K.expand_dims(a, axis=-1)  # [batch_size,JX,JQ,1]
            e_i = multiply([e_t, targets])  # [batch_size,JX,JQ,d]
            return K.sum(e_i, axis=-2)
        else:
            return K.batch_dot(a, targets)

    def call(self, inputs, mask=None):  # question based on the parameters VrQ
        h = inputs  # context [batch,P,2d]

        if self.linear_fun != 'tri':
            return super(FasterSelf_Attention4BiDAF).call(inputs, mask=mask)

        self.JX = K.shape(h)[1]

        u_logits = self._get_logits(h, h)
        
        if mask is not None:  # necessary for long length input
            h_mask = mask
            # mask need expand
            h_mask_aug = K.tile(K.expand_dims(h_mask, axis=2), [1, 1, self.JX])
            u_mask_aug = K.tile(K.expand_dims(h_mask, axis=1), [1, self.JX, 1])
            hu_mask = h_mask_aug & u_mask_aug  # mask是都
            # diagonal_mask
            diag_ones = K.ones_like(h_mask_aug, dtype='float32')
            diag_zeros = K.zeros_like(h_mask_aug, dtype='float32')
            diag = tf.matrix_diag_part(diag_ones)
            diag_mask = tf.matrix_set_diag(diag_zeros, diag)
            hud_mask = hu_mask & K.cast(diag_mask, 'bool')
            # add mask
            u_logits = exp_mask(u_logits, hud_mask)

        u_a = self._softsel(h, u_logits)
        p0 = concatenate([h, u_a, h * u_a])  # 6d*T

        return p0
