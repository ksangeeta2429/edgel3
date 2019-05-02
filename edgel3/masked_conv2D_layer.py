from keras.layers import Layer
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import dtypes


class MaskedConv2D(Layer):
  def __init__(self,
               threshold,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):

    super(MaskedConv2D, self).__init__(trainable=trainable,
                                       name=name,
                                       **kwargs)

    self.rank = rank
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank,
                                                    'dilation_rate')
    
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = InputSpec(ndim=self.rank + 2)
    self.threshold = threshold


  def build(self, input_shape):
    #input_shape = tensor_shape.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    self.kernel = self.add_weight(name='kernel',
                                  shape=kernel_shape,
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint,
                                  trainable=True,
                                  dtype=dtypes.float32)

                                      
    self.mask = K.cast(K.greater(K.abs(self.kernel), self.threshold), dtypes.float32)
    self.masked_kernel = math_ops.multiply(self.mask, self.kernel)

    if self.use_bias:
      self.bias = self.add_weight(name='bias',
                                  shape=(self.filters,),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  trainable=True,
                                  dtype=dtypes.float32)
    else:
      self.bias = None

    self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
   
    '''
    self._convolution_op = nn_ops.Convolution(input_shape,
                                              filter_shape=self.kernel.get_shape(),
                                              dilation_rate=self.dilation_rate,
                                              strides=self.strides,
                                              padding=op_padding.upper(),
                                              data_format=conv_utils.convert_data_format(self.data_format, self.rank + 2))
    '''
    self.built = True

  def call(self, inputs):
    #outputs = self._convolution_op(inputs, self.masked_kernel)
    if self.rank == 1:
      outputs = K.conv1d(
                inputs,
                self.masked_kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
    if self.rank == 2:
      outputs = K.conv2d(
                inputs,
                self.masked_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
    if self.rank == 3:
      outputs = K.conv3d(
                inputs,
                self.masked_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    
    return outputs


  def compute_output_shape(self, input_shape):
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(space[i],
                                                self.kernel_size[i],
                                                padding=self.padding,
                                                stride=self.strides[i],
                                                dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return (input_shape[0],) + tuple(new_space) + (self.filters,)

  def get_config(self):
    config = {
      'threshold': self.threshold,
      'rank': self.rank,
      'filters': self.filters,
      'kernel_size': self.kernel_size,
      'strides': self.strides,
      'padding': self.padding,
      'data_format': self.data_format,
      'dilation_rate': self.dilation_rate,
      'activation': activations.serialize(self.activation),
      'use_bias': self.use_bias,
      'kernel_initializer': initializers.serialize(self.kernel_initializer),
      'bias_initializer': initializers.serialize(self.bias_initializer),
      'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer': regularizers.serialize(self.bias_regularizer),
      'activity_regularizer': regularizers.serialize(self.activity_regularizer),
      'kernel_constraint': constraints.serialize(self.kernel_constraint),
      'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(MaskedConv2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
