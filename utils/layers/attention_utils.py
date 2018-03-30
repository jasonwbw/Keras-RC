
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
