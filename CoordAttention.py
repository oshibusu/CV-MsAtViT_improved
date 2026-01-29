from keras.layers import  Lambda, concatenate

from cvnn.layers import ComplexConv2D, ComplexAvgPooling2D, ComplexBatchNormalization

from cvnn.activations import cart_relu
from keras import backend as K
import tensorflow as tf
from SAR_utils import ModTanhScaled, ModSigmoid, cart_sigmoid


class ComplexSplit(tf.keras.layers.Layer):
    def __init__(self, num_or_size_splits, axis=0, **kwargs):
        super(ComplexSplit, self).__init__(**kwargs)
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis

    def call(self, inputs):
        return tf.split(inputs, num_or_size_splits=self.num_or_size_splits, axis=self.axis)

    def get_config(self):
        config = super(ComplexSplit, self).get_config()
        config.update({
            "num_or_size_splits": self.num_or_size_splits,
            "axis": self.axis,
        })
        return config


def CoordAtt_cmplx(x, reduction = 8, activation="modtanh"):

    def coord_act(x):
        tmpx = cart_relu((x + 3), max_value=6) / 6
        x = x * tmpx
        return x

    x_shape = x.shape.as_list()
    [b, h, w, c] = x_shape
    x_h = ComplexAvgPooling2D(pool_size=(1, w), strides=(1, 1), data_format='channels_last')(x)
    x_w = ComplexAvgPooling2D(pool_size=(h, 1), strides=(1, 1), data_format='channels_last')(x)
    x_w = K.permute_dimensions(x_w, [0, 2, 1, 3])
    y = concatenate(inputs=[x_h, x_w], axis=1)
    mip = max(8, c // reduction)
    y = ComplexConv2D(filters=mip, kernel_size=(1, 1), strides=(1, 1), padding='valid')(y)
    y = ComplexBatchNormalization(trainable=False)(y)
    y = coord_act(y)
    
    x_h, x_w = ComplexSplit(num_or_size_splits=[h, w], axis=1)(y)
    
    x_w = K.permute_dimensions(x_w, [0, 2, 1, 3])
    a_h = ComplexConv2D(filters=c, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation=None)(x_h)
    
    # Apply activation based on type
    if activation == "modtanh":
        a_h = ModTanhScaled()(a_h)
    elif activation == "modsigmoid":
        a_h = ModSigmoid()(a_h)
    elif activation == "cart_sigmoid":
        a_h = Lambda(cart_sigmoid)(a_h)
    
    a_w = ComplexConv2D(filters=c, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation=None)(x_w)
    
    if activation == "modtanh":
        a_w = ModTanhScaled()(a_w)
    elif activation == "modsigmoid":
        a_w = ModSigmoid()(a_w)
    elif activation == "cart_sigmoid":
        a_w = Lambda(cart_sigmoid)(a_w)
        
    out = x * (a_h * a_w)
    return out

