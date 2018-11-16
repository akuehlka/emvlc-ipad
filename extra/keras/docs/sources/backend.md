# Keras backends

## What is a "backend"?

Keras is a model-level library, providing high-level building blocks for developing deep learning models. It does not handle itself low-level operations such as tensor products, convolutions and so on. Instead, it relies on a specialized, well-optimized tensor manipulation library to do so, serving as the "backend engine" of Keras. Rather than picking one single tensor library and making the implementation of Keras tied to that library, Keras handles the problem in a modular way, and several different backend engines can be plugged seamlessly into Keras.

At this time, Keras has three backend implementations available: the **TensorFlow** backend, the **Theano** backend, and the **CNTK** backend.

- [TensorFlow](http://www.tensorflow.org/) is an open-source symbolic tensor manipulation framework developed by Google, Inc.
- [Theano](http://deeplearning.net/software/theano/) is an open-source symbolic tensor manipulation framework developed by LISA/MILA Lab at Université de Montréal.
- [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/) is an open-source, commercial-grade toolkit for deep learning developed by Microsoft.

In the future, we are likely to add more backend options.

----

## Switching from one backend to another

If you have run Keras at least once, you will find the Keras configuration file at:

`$HOME/.keras/keras.json`

If it isn't there, you can create it.

**NOTE for Windows Users:** Please change `$HOME` with `%USERPROFILE%`.

The default configuration file looks like this:

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

Simply change the field `backend` to `"theano"`, `"tensorflow"`, or `"cntk"`, and Keras will use the new configuration next time you run any Keras code.

You can also define the environment variable ``KERAS_BACKEND`` and this will
override what is defined in your config file :

```bash
KERAS_BACKEND=tensorflow python -c "from keras import backend"
Using TensorFlow backend.
```

----

## keras.json details


```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

You can change these settings by editing `$HOME/.keras/keras.json`. 

* `image_data_format`: string, either `"channels_last"` or `"channels_first"`. It specifies which data format convention Keras will follow. (`keras.backend.image_data_format()` returns it.)
  - For 2D data (e.g. image), `"channels_last"` assumes `(rows, cols, channels)` while `"channels_first"` assumes `(channels, rows, cols)`. 
  - For 3D data, `"channels_last"` assumes `(conv_dim1, conv_dim2, conv_dim3, channels)` while `"channels_first"` assumes `(channels, conv_dim1, conv_dim2, conv_dim3)`.
* `epsilon`: float, a numeric fuzzing constant used to avoid dividing by zero in some operations.
* `floatx`: string, `"float16"`, `"float32"`, or `"float64"`. Default float precision.
* `backend`: string, `"tensorflow"`, `"theano"`, or `"cntk"`.

----

## Using the abstract Keras backend to write new code

If you want the Keras modules you write to be compatible with both Theano (`th`) and TensorFlow (`tf`), you have to write them via the abstract Keras backend API. Here's an intro.

You can import the backend module via:
```python
from keras import backend as K
```

The code below instantiates an input placeholder. It's equivalent to `tf.placeholder()` or `th.tensor.matrix()`, `th.tensor.tensor3()`, etc.

```python
input = K.placeholder(shape=(2, 4, 5))
# also works:
input = K.placeholder(shape=(None, 4, 5))
# also works:
input = K.placeholder(ndim=3)
```

The code below instantiates a shared variable. It's equivalent to `tf.Variable()` or `th.shared()`.

```python
import numpy as np
val = np.random.random((3, 4, 5))
var = K.variable(value=val)

# all-zeros variable:
var = K.zeros(shape=(3, 4, 5))
# all-ones:
var = K.ones(shape=(3, 4, 5))
```

Most tensor operations you will need can be done as you would in TensorFlow or Theano:

```python
# Initializing Tensors with Random Numbers
b = K.random_uniform_variable(shape=(3, 4)). # Uniform distribution
c = K.random_normal_variable(shape=(3, 4)). # Gaussian distribution
d = K.random_normal_variable(shape=(3, 4)).
# Tensor Arithmetics
a = b + c * K.abs(d)
c = K.dot(a, K.transpose(b))
a = K.sum(b, axis=1)
a = K.softmax(b)
a = K.concatenate([b, c], axis=-1)
# etc...
```

----

## Backend functions


### backend


```python
backend()
```


Publicly accessible method
for determining the current backend.

__Returns__

String, the name of the backend Keras is currently using.

__Example__

```python
>>> keras.backend.backend()
'tensorflow'
```

----

### epsilon


```python
epsilon()
```


Returns the value of the fuzz
factor used in numeric expressions.

__Returns__

A float.

__Example__

```python
>>> keras.backend.epsilon()
1e-08
```

----

### set_epsilon


```python
set_epsilon(e)
```


Sets the value of the fuzz
factor used in numeric expressions.

__Arguments__

- __e__: float. New value of epsilon.

__Example__

```python
>>> from keras import backend as K
>>> K.epsilon()
1e-08
>>> K.set_epsilon(1e-05)
>>> K.epsilon()
1e-05
```

----

### floatx


```python
floatx()
```


Returns the default float type, as a string.
(e.g. 'float16', 'float32', 'float64').

__Returns__

String, the current default float type.

__Example__

```python
>>> keras.backend.floatx()
'float32'
```

----

### set_floatx


```python
set_floatx(floatx)
```


Sets the default float type.

__Arguments__

- __String__: 'float16', 'float32', or 'float64'.

__Example__

```python
>>> from keras import backend as K
>>> K.floatx()
'float32'
>>> K.set_floatx('float16')
>>> K.floatx()
'float16'
```

----

### cast_to_floatx


```python
cast_to_floatx(x)
```


Cast a Numpy array to the default Keras float type.

__Arguments__

- __x__: Numpy array.

__Returns__

The same Numpy array, cast to its new type.

__Example__

```python
>>> from keras import backend as K
>>> K.floatx()
'float32'
>>> arr = numpy.array([1.0, 2.0], dtype='float64')
>>> arr.dtype
dtype('float64')
>>> new_arr = K.cast_to_floatx(arr)
>>> new_arr
array([ 1.,  2.], dtype=float32)
>>> new_arr.dtype
dtype('float32')
```

----

### image_data_format


```python
image_data_format()
```


Returns the default image data format convention ('channels_first' or 'channels_last').

__Returns__

A string, either `'channels_first'` or `'channels_last'`

__Example__

```python
>>> keras.backend.image_data_format()
'channels_first'
```

----

### set_image_data_format


```python
set_image_data_format(data_format)
```


Sets the value of the data format convention.

__Arguments__

- __data_format__: string. `'channels_first'` or `'channels_last'`.

__Example__

```python
>>> from keras import backend as K
>>> K.image_data_format()
'channels_first'
>>> K.set_image_data_format('channels_last')
>>> K.image_data_format()
'channels_last'
```

----

### set_image_dim_ordering


```python
set_image_dim_ordering(dim_ordering)
```


Legacy setter for `image_data_format`.

__Arguments__

- __dim_ordering__: string. `tf` or `th`.

__Example__

```python
>>> from keras import backend as K
>>> K.image_data_format()
'channels_first'
>>> K.set_image_data_format('channels_last')
>>> K.image_data_format()
'channels_last'
```

__Raises__

- __ValueError__: if `dim_ordering` is invalid.

----

### image_dim_ordering


```python
image_dim_ordering()
```


Legacy getter for `image_data_format`.

__Returns__

string, one of `'th'`, `'tf'`

----

### get_uid


```python
get_uid(prefix='')
```


Get the uid for the default graph.

__Arguments__

- __prefix__: An optional prefix of the graph.

__Returns__

A unique identifier for the graph.

----

### reset_uids


```python
reset_uids()
```


Reset graph identifiers.
----

### clear_session


```python
clear_session()
```


Destroys the current TF graph and creates a new one.

Useful to avoid clutter from old models / layers.

----

### manual_variable_initialization


```python
manual_variable_initialization(value)
```


Sets the manual variable initialization flag.

This boolean flag determines whether
variables should be initialized
as they are instantiated (default), or if
the user should handle the initialization
(e.g. via `tf.initialize_all_variables()`).

__Arguments__

- __value__: Python boolean.

----

### learning_phase


```python
learning_phase()
```


Returns the learning phase flag.

The learning phase flag is a bool tensor (0 = test, 1 = train)
to be passed as input to any Keras function
that uses a different behavior at train time and test time.

__Returns__

Learning phase (scalar integer tensor or Python integer).

----

### set_learning_phase


```python
set_learning_phase(value)
```


Sets the learning phase to a fixed value.

__Arguments__

- __value__: Learning phase value, either 0 or 1 (integers).

__Raises__

- __ValueError__: if `value` is neither `0` nor `1`.

----

### is_sparse


```python
is_sparse(tensor)
```


Returns whether a tensor is a sparse tensor.

__Arguments__

- __tensor__: A tensor instance.

__Returns__

A boolean.

__Example__

```python
>>> from keras import backend as K
>>> a = K.placeholder((2, 2), sparse=False)
>>> print(K.is_sparse(a))
False
>>> b = K.placeholder((2, 2), sparse=True)
>>> print(K.is_sparse(b))
True
```

----

### to_dense


```python
to_dense(tensor)
```


Converts a sparse tensor into a dense tensor and returns it.

__Arguments__

- __tensor__: A tensor instance (potentially sparse).

__Returns__

A dense tensor.

__Examples__

```python
>>> from keras import backend as K
>>> b = K.placeholder((2, 2), sparse=True)
>>> print(K.is_sparse(b))
True
>>> c = K.to_dense(b)
>>> print(K.is_sparse(c))
False
```

----

### variable


```python
variable(value, dtype=None, name=None)
```


Instantiates a variable and returns it.

__Arguments__

- __value__: Numpy array, initial value of the tensor.
- __dtype__: Tensor type.
- __name__: Optional name string for the tensor.

__Returns__

A variable instance (with Keras metadata included).

__Examples__

```python
>>> from keras import backend as K
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val, dtype='float64', name='example_var')
>>> K.dtype(kvar)
'float64'
>>> print(kvar)
example_var
>>> kvar.eval()
array([[ 1.,  2.],
   [ 3.,  4.]])
```

----

### constant


```python
constant(value, dtype=None, shape=None, name=None)
```


Creates a constant tensor.

__Arguments__

- __value__: A constant value (or list)
- __dtype__: The type of the elements of the resulting tensor.
- __shape__: Optional dimensions of resulting tensor.
- __name__: Optional name for the tensor.

__Returns__

A Constant Tensor.

----

### is_keras_tensor


```python
is_keras_tensor(x)
```


Returns whether `x` is a Keras tensor.

__Arguments__

- __x__: a potential tensor.

__Returns__

A boolean: whether the argument is a Keras tensor.

__Raises__

- __ValueError__: in case `x` is not a symbolic tensor.

__Examples__

```python
>>> from keras import backend as K
>>> np_var = numpy.array([1, 2])
>>> K.is_keras_tensor(np_var) # A numpy array is not a symbolic yensor.
ValueError
>>> k_var = tf.placeholder('float32', shape=(1,1))
>>> K.is_keras_tensor(k_var) # A variable created directly from tensorflow/theano is not a Keras tensor.
False
>>> keras_var = K.variable(np_var)
>>> K.is_keras_tensor(keras_var)  # A variable created with the keras backend is a Keras tensor.
True
>>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
>>> K.is_keras_tensor(keras_placeholder)  # A placeholder is a Keras tensor.
True
```

----

### placeholder


```python
placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None)
```


Instantiates a placeholder tensor and returns it.

__Arguments__

- __shape__: Shape of the placeholder
(integer tuple, may include `None` entries).
- __ndim__: Number of axes of the tensor.
At least one of {`shape`, `ndim`} must be specified.
If both are specified, `shape` is used.
- __dtype__: Placeholder type.
- __sparse__: Boolean, whether the placeholder should have a sparse type.
- __name__: Optional name string for the placeholder.

__Returns__

Tensor instance (with Keras metadata included).

__Examples__

```python
>>> from keras import backend as K
>>> input_ph = K.placeholder(shape=(2, 4, 5))
>>> input_ph._keras_shape
(2, 4, 5)
>>> input_ph
<tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
```

----

### shape


```python
shape(x)
```


Returns the symbolic shape of a tensor or variable.

__Arguments__

- __x__: A tensor or variable.

__Returns__

A symbolic shape (which is itself a tensor).

__Examples__

```
__TensorFlow example__

>>> from keras import backend as K
>>> tf_session = K.get_session()
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> input = keras.backend.placeholder(shape=(2, 4, 5))
>>> K.shape(kvar)
<tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
>>> K.shape(input)
<tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
__To get integer shape (Instead, you can use K.int_shape(x))__

>>> K.shape(kvar).eval(session=tf_session)
array([2, 2], dtype=int32)
>>> K.shape(input).eval(session=tf_session)
array([2, 4, 5], dtype=int32)
```

----

### int_shape


```python
int_shape(x)
```


Returns the shape tensor or variable as a tuple of int or None entries.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tuple of integers (or None entries).

__Examples__

```python
>>> from keras import backend as K
>>> input = K.placeholder(shape=(2, 4, 5))
>>> K.int_shape(input)
(2, 4, 5)
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> K.int_shape(kvar)
(2, 2)
```

----

### ndim


```python
ndim(x)
```


Returns the number of axes in a tensor, as an integer.

__Arguments__

- __x__: Tensor or variable.

__Returns__

Integer (scalar), number of axes.

__Examples__

```python
>>> from keras import backend as K
>>> input = K.placeholder(shape=(2, 4, 5))
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> K.ndim(input)
3
>>> K.ndim(kvar)
2
```

----

### switch


```python
switch(condition, then_expression, else_expression)
```


Switches between two operations depending on a scalar value.

Note that both `then_expression` and `else_expression`
should be symbolic tensors of the *same shape*.

__Arguments__

- __condition__: scalar tensor (`int` or `bool`).
- __then_expression__: either a tensor, or a callable that returns a tensor.
- __else_expression__: either a tensor, or a callable that returns a tensor.

__Returns__

The selected tensor.

----

### in_train_phase


```python
in_train_phase(x, alt, training=None)
```


Selects `x` in train phase, and `alt` otherwise.

Note that `alt` should have the *same shape* as `x`.

__Arguments__

- __x__: What to return in train phase
(tensor or callable that returns a tensor).
- __alt__: What to return otherwise
(tensor or callable that returns a tensor).
- __training__: Optional scalar tensor
(or Python boolean, or Python integer)
specifing the learning phase.

__Returns__

Either `x` or `alt` based on the `training` flag.
the `training` flag defaults to `K.learning_phase()`.

----

### in_test_phase


```python
in_test_phase(x, alt, training=None)
```


Selects `x` in test phase, and `alt` otherwise.

Note that `alt` should have the *same shape* as `x`.

__Arguments__

- __x__: What to return in test phase
(tensor or callable that returns a tensor).
- __alt__: What to return otherwise
(tensor or callable that returns a tensor).
- __training__: Optional scalar tensor
(or Python boolean, or Python integer)
specifing the learning phase.

__Returns__

Either `x` or `alt` based on `K.learning_phase`.

----

### relu


```python
relu(x, alpha=0.0, max_value=None)
```


Rectified linear unit.

With default values, it returns element-wise `max(x, 0)`.

__Arguments__

- __x__: A tensor or variable.
- __alpha__: A scalar, slope of negative section (default=`0.`).
- __max_value__: Saturation threshold.

__Returns__

A tensor.

----

### elu


```python
elu(x, alpha=1.0)
```


Exponential linear unit.

__Arguments__

- __x__: A tenor or variable to compute the activation function for.
- __alpha__: A scalar, slope of positive section.

__Returns__

A tensor.

----

### softmax


```python
softmax(x)
```


Softmax of a tensor.

__Arguments__

- __x__: A tensor or variable.

__Returns__

A tensor.

----

### softplus


```python
softplus(x)
```


Softplus of a tensor.

__Arguments__

- __x__: A tensor or variable.

__Returns__

A tensor.

----

### softsign


```python
softsign(x)
```


Softsign of a tensor.

__Arguments__

- __x__: A tensor or variable.

__Returns__

A tensor.

----

### categorical_crossentropy


```python
categorical_crossentropy(output, target, from_logits=False)
```


Categorical crossentropy between an output tensor and a target tensor.

__Arguments__

- __output__: A tensor resulting from a softmax
(unless `from_logits` is True, in which
case `output` is expected to be the logits).
- __target__: A tensor of the same shape as `output`.
- __from_logits__: Boolean, whether `output` is the
result of a softmax, or is a tensor of logits.

__Returns__

Output tensor.

----

### sparse_categorical_crossentropy


```python
sparse_categorical_crossentropy(output, target, from_logits=False)
```


Categorical crossentropy with integer targets.

__Arguments__

- __output__: A tensor resulting from a softmax
(unless `from_logits` is True, in which
case `output` is expected to be the logits).
- __target__: An integer tensor.
- __from_logits__: Boolean, whether `output` is the
result of a softmax, or is a tensor of logits.

__Returns__

Output tensor.

----

### binary_crossentropy


```python
binary_crossentropy(output, target, from_logits=False)
```


Binary crossentropy between an output tensor and a target tensor.

__Arguments__

- __output__: A tensor.
- __target__: A tensor with the same shape as `output`.
- __from_logits__: Whether `output` is expected to be a logits tensor.
By default, we consider that `output`
encodes a probability distribution.

__Returns__

A tensor.

----

### sigmoid


```python
sigmoid(x)
```


Element-wise sigmoid.

__Arguments__

- __x__: A tensor or variable.

__Returns__

A tensor.

----

### hard_sigmoid


```python
hard_sigmoid(x)
```


Segment-wise linear approximation of sigmoid.

Faster than sigmoid.
Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

__Arguments__

- __x__: A tensor or variable.

__Returns__

A tensor.

----

### tanh


```python
tanh(x)
```


Element-wise tanh.

__Arguments__

- __x__: A tensor or variable.

__Returns__

A tensor.

----

### dropout


```python
dropout(x, level, noise_shape=None, seed=None)
```


Sets entries in `x` to zero at random, while scaling the entire tensor.

__Arguments__

- __x__: tensor
- __level__: fraction of the entries in the tensor
that will be set to 0.
- __noise_shape__: shape for randomly generated keep/drop flags,
must be broadcastable to the shape of `x`
- __seed__: random seed to ensure determinism.

__Returns__

A tensor.

----

### l2_normalize


```python
l2_normalize(x, axis)
```


Normalizes a tensor wrt the L2 norm alongside the specified axis.

__Arguments__

- __x__: Tensor or variable.
- __axis__: axis along which to perform normalization.

__Returns__

A tensor.

----

### in_top_k


```python
in_top_k(predictions, targets, k)
```


Returns whether the `targets` are in the top `k` `predictions`.

__Arguments__

- __predictions__: A tensor of shape `(batch_size, classes)` and type `float32`.
- __targets__: A 1D tensor of length `batch_size` and type `int32` or `int64`.
- __k__: An `int`, number of top elements to consider.

__Returns__

A 1D tensor of length `batch_size` and type `bool`.
`output[i]` is `True` if `predictions[i, targets[i]]` is within top-`k`
values of `predictions[i]`.

----

### conv1d


```python
conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1)
```


1D convolution.

__Arguments__

- __x__: Tensor or variable.
- __kernel__: kernel tensor.
- __strides__: stride integer.
- __padding__: string, `"same"`, `"causal"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __dilation_rate__: integer dilate rate.

__Returns__

A tensor, result of 1D convolution.

----

### conv2d


```python
conv2d(x, kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


2D convolution.

__Arguments__

- __x__: Tensor or variable.
- __kernel__: kernel tensor.
- __strides__: strides tuple.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
Whether to use Theano or TensorFlow data format
for inputs/kernels/outputs.
- __dilation_rate__: tuple of 2 integers.

__Returns__

A tensor, result of 2D convolution.

__Raises__

- __ValueError__: if `data_format` is neither `channels_last` or `channels_first`.

----

### conv2d_transpose


```python
conv2d_transpose(x, kernel, output_shape, strides=(1, 1), padding='valid', data_format=None)
```


2D deconvolution (i.e. transposed convolution).

__Arguments__

- __x__: Tensor or variable.
- __kernel__: kernel tensor.
- __output_shape__: 1D int tensor for the output shape.
- __strides__: strides tuple.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
Whether to use Theano or TensorFlow data format
for inputs/kernels/outputs.

__Returns__

A tensor, result of transposed 2D convolution.

__Raises__

- __ValueError__: if `data_format` is neither `channels_last` or `channels_first`.

----

### separable_conv2d


```python
separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


2D convolution with separable filters.

__Arguments__

- __x__: input tensor
- __depthwise_kernel__: convolution kernel for the depthwise convolution.
- __pointwise_kernel__: kernel for the 1x1 convolution.
- __strides__: strides tuple (length 2).
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __dilation_rate__: tuple of integers,
dilation rates for the separable convolution.

__Returns__

Output tensor.

__Raises__

- __ValueError__: if `data_format` is neither `channels_last` or `channels_first`.

----

### mean


```python
mean(x, axis=None, keepdims=False)
```


Mean of a tensor, alongside the specified axis.

__Arguments__

- __x__: A tensor or variable.
- __axis__: A list of integer. Axes to compute the mean.
- __keepdims__: A boolean, whether to keep the dimensions or not.
If `keepdims` is `False`, the rank of the tensor is reduced
by 1 for each entry in `axis`. If `keep_dims` is `True`,
the reduced dimensions are retained with length 1.

__Returns__

A tensor with the mean of elements of `x`.

----

### any


```python
any(x, axis=None, keepdims=False)
```


Bitwise reduction (logical OR).

__Arguments__

- __x__: Tensor or variable.
- __axis__: axis along which to perform the reduction.
- __keepdims__: whether the drop or broadcast the reduction axes.

__Returns__

A uint8 tensor (0s and 1s).

----

### all


```python
all(x, axis=None, keepdims=False)
```


Bitwise reduction (logical AND).

__Arguments__

- __x__: Tensor or variable.
- __axis__: axis along which to perform the reduction.
- __keepdims__: whether the drop or broadcast the reduction axes.

__Returns__

A uint8 tensor (0s and 1s).

----

### argmax


```python
argmax(x, axis=-1)
```


Returns the index of the maximum value along an axis.

__Arguments__

- __x__: Tensor or variable.
- __axis__: axis along which to perform the reduction.

__Returns__

A tensor.

----

### argmin


```python
argmin(x, axis=-1)
```


Returns the index of the minimum value along an axis.

__Arguments__

- __x__: Tensor or variable.
- __axis__: axis along which to perform the reduction.

__Returns__

A tensor.

----

### square


```python
square(x)
```


Element-wise square.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.

----

### abs


```python
abs(x)
```


Element-wise absolute value.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.

----

### sqrt


```python
sqrt(x)
```


Element-wise square root.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.

----

### exp


```python
exp(x)
```


Element-wise exponential.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.

----

### log


```python
log(x)
```


Element-wise log.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.

----

### logsumexp


```python
logsumexp(x, axis=None, keepdims=False)
```


Computes log(sum(exp(elements across dimensions of a tensor))).

This function is more numerically stable than log(sum(exp(x))).
It avoids overflows caused by taking the exp of large inputs and
underflows caused by taking the log of small inputs.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer, the axis to reduce over.
- __keepdims__: A boolean, whether to keep the dimensions or not.
If `keepdims` is `False`, the rank of the tensor is reduced
by 1. If `keepdims` is `True`, the reduced dimension is
retained with length 1.

__Returns__

The reduced tensor.

----

### round


```python
round(x)
```


Element-wise rounding to the closest integer.

In case of tie, the rounding mode used is "half to even".

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.

----

### sign


```python
sign(x)
```


Element-wise sign.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.

----

### pow


```python
pow(x, a)
```


Element-wise exponentiation.

__Arguments__

- __x__: Tensor or variable.
- __a__: Python integer.

__Returns__

A tensor.

----

### clip


```python
clip(x, min_value, max_value)
```


Element-wise value clipping.

__Arguments__

- __x__: Tensor or variable.
- __min_value__: Python float or integer.
- __max_value__: Python float or integer.

__Returns__

A tensor.

----

### equal


```python
equal(x, y)
```


Element-wise equality between two tensors.

__Arguments__

- __x__: Tensor or variable.
- __y__: Tensor or variable.

__Returns__

A bool tensor.

----

### not_equal


```python
not_equal(x, y)
```


Element-wise inequality between two tensors.

__Arguments__

- __x__: Tensor or variable.
- __y__: Tensor or variable.

__Returns__

A bool tensor.

----

### greater


```python
greater(x, y)
```


Element-wise truth value of (x > y).

__Arguments__

- __x__: Tensor or variable.
- __y__: Tensor or variable.

__Returns__

A bool tensor.

----

### greater_equal


```python
greater_equal(x, y)
```


Element-wise truth value of (x >= y).

__Arguments__

- __x__: Tensor or variable.
- __y__: Tensor or variable.

__Returns__

A bool tensor.

----

### less


```python
less(x, y)
```


Element-wise truth value of (x < y).

__Arguments__

- __x__: Tensor or variable.
- __y__: Tensor or variable.

__Returns__

A bool tensor.

----

### less_equal


```python
less_equal(x, y)
```


Element-wise truth value of (x <= y).

__Arguments__

- __x__: Tensor or variable.
- __y__: Tensor or variable.

__Returns__

A bool tensor.

----

### maximum


```python
maximum(x, y)
```


Element-wise maximum of two tensors.

__Arguments__

- __x__: Tensor or variable.
- __y__: Tensor or variable.

__Returns__

A tensor.

----

### minimum


```python
minimum(x, y)
```


Element-wise minimum of two tensors.

__Arguments__

- __x__: Tensor or variable.
- __y__: Tensor or variable.

__Returns__

A tensor.

----

### sin


```python
sin(x)
```


Computes sin of x element-wise.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.

----

### cos


```python
cos(x)
```


Computes cos of x element-wise.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.

----

### normalize_batch_in_training


```python
normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=0.001)
```


Computes mean and std for batch then apply batch_normalization on batch.

__Arguments__

- __x__: Input tensor or variable.
- __gamma__: Tensor by which to scale the input.
- __beta__: Tensor with which to center the input.
- __reduction_axes__: iterable of integers,
axes over which to normalize.
- __epsilon__: Fuzz factor.

__Returns__

A tuple length of 3, `(normalized_tensor, mean, variance)`.

----

### batch_normalization


```python
batch_normalization(x, mean, var, beta, gamma, epsilon=0.001)
```


Applies batch normalization on x given mean, var, beta and gamma.

I.e. returns:
`output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`

__Arguments__

- __x__: Input tensor or variable.
- __mean__: Mean of batch.
- __var__: Variance of batch.
- __beta__: Tensor with which to center the input.
- __gamma__: Tensor by which to scale the input.
- __epsilon__: Fuzz factor.

__Returns__

A tensor.

----

### concatenate


```python
concatenate(tensors, axis=-1)
```


Concatenates a list of tensors alongside the specified axis.

__Arguments__

- __tensors__: list of tensors to concatenate.
- __axis__: concatenation axis.

__Returns__

A tensor.

----

### reshape


```python
reshape(x, shape)
```


Reshapes a tensor to the specified shape.

__Arguments__

- __x__: Tensor or variable.
- __shape__: Target shape tuple.

__Returns__

A tensor.

----

### dtype


```python
dtype(x)
```


Returns the dtype of a Keras tensor or variable, as a string.

__Arguments__

- __x__: Tensor or variable.

__Returns__

String, dtype of `x`.

__Examples__

```python
>>> from keras import backend as K
>>> K.dtype(K.placeholder(shape=(2,4,5)))
'float32'
>>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
'float32'
>>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
'float64'
__Keras variable__

>>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
>>> K.dtype(kvar)
'float32_ref'
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
>>> K.dtype(kvar)
'float32_ref'
```

----

### eval


```python
eval(x)
```


Evaluates the value of a variable.

__Arguments__

- __x__: A variable.

__Returns__

A Numpy array.

__Examples__

```python
>>> from keras import backend as K
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
>>> K.eval(kvar)
array([[ 1.,  2.],
   [ 3.,  4.]], dtype=float32)
```

----

### zeros


```python
zeros(shape, dtype=None, name=None)
```


Instantiates an all-zeros variable and returns it.

__Arguments__

- __shape__: Tuple of integers, shape of returned Keras variable
- __dtype__: String, data type of returned Keras variable
- __name__: String, name of returned Keras variable

__Returns__

A variable (including Keras metadata), filled with `0.0`.

__Example__

```python
>>> from keras import backend as K
>>> kvar = K.zeros((3,4))
>>> K.eval(kvar)
array([[ 0.,  0.,  0.,  0.],
   [ 0.,  0.,  0.,  0.],
   [ 0.,  0.,  0.,  0.]], dtype=float32)
```

----

### ones


```python
ones(shape, dtype=None, name=None)
```


Instantiates an all-ones tensor variable and returns it.

__Arguments__

- __shape__: Tuple of integers, shape of returned Keras variable.
- __dtype__: String, data type of returned Keras variable.
- __name__: String, name of returned Keras variable.

__Returns__

A Keras variable, filled with `1.0`.

__Example__

```python
>>> from keras import backend as K
>>> kvar = K.ones((3,4))
>>> K.eval(kvar)
array([[ 1.,  1.,  1.,  1.],
   [ 1.,  1.,  1.,  1.],
   [ 1.,  1.,  1.,  1.]], dtype=float32)
```

----

### eye


```python
eye(size, dtype=None, name=None)
```


Instantiate an identity matrix and returns it.

__Arguments__

- __size__: Integer, number of rows/columns.
- __dtype__: String, data type of returned Keras variable.
- __name__: String, name of returned Keras variable.

__Returns__

A Keras variable, an identity matrix.

__Example__

```python
>>> from keras import backend as K
>>> kvar = K.eye(3)
>>> K.eval(kvar)
array([[ 1.,  0.,  0.],
   [ 0.,  1.,  0.],
   [ 0.,  0.,  1.]], dtype=float32)
```


----

### zeros_like


```python
zeros_like(x, dtype=None, name=None)
```


Instantiates an all-zeros variable of the same shape as another tensor.

__Arguments__

- __x__: Keras variable or Keras tensor.
- __dtype__: String, dtype of returned Keras variable.
 None uses the dtype of x.
- __name__: String, name for the variable to create.

__Returns__

A Keras variable with the shape of x filled with zeros.

__Example__

```python
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_zeros = K.zeros_like(kvar)
>>> K.eval(kvar_zeros)
array([[ 0.,  0.,  0.],
   [ 0.,  0.,  0.]], dtype=float32)
```

----

### ones_like


```python
ones_like(x, dtype=None, name=None)
```


Instantiates an all-ones variable of the same shape as another tensor.

__Arguments__

- __x__: Keras variable or tensor.
- __dtype__: String, dtype of returned Keras variable.
 None uses the dtype of x.
- __name__: String, name for the variable to create.

__Returns__

A Keras variable with the shape of x filled with ones.

__Example__

```python
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_ones = K.ones_like(kvar)
>>> K.eval(kvar_ones)
array([[ 1.,  1.,  1.],
   [ 1.,  1.,  1.]], dtype=float32)
```

----

### identity


```python
identity(x)
```


Returns a tensor with the same content as the input tensor.

__Arguments__

- __x__: The input tensor.

__Returns__

A tensor of the same shape, type and content.

----

### random_uniform_variable


```python
random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None)
```


Instantiates a variable with values drawn from a uniform distribution.

__Arguments__

- __shape__: Tuple of integers, shape of returned Keras variable.
- __low__: Float, lower boundary of the output interval.
- __high__: Float, upper boundary of the output interval.
- __dtype__: String, dtype of returned Keras variable.
- __name__: String, name of returned Keras variable.
- __seed__: Integer, random seed.

__Returns__

A Keras variable, filled with drawn samples.

__Example__

```python
__TensorFlow example__

>>> kvar = K.random_uniform_variable((2,3), 0, 1)
>>> kvar
<tensorflow.python.ops.variables.Variable object at 0x10ab40b10>
>>> K.eval(kvar)
array([[ 0.10940075,  0.10047495,  0.476143  ],
   [ 0.66137183,  0.00869417,  0.89220798]], dtype=float32)
```

----

### random_normal_variable


```python
random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None)
```


Instantiates a variable with values drawn from a normal distribution.

__Arguments__

- __shape__: Tuple of integers, shape of returned Keras variable.
- __mean__: Float, mean of the normal distribution.
- __scale__: Float, standard deviation of the normal distribution.
- __dtype__: String, dtype of returned Keras variable.
- __name__: String, name of returned Keras variable.
- __seed__: Integer, random seed.

__Returns__

A Keras variable, filled with drawn samples.

__Example__

```python
__TensorFlow example__

>>> kvar = K.random_normal_variable((2,3), 0, 1)
>>> kvar
<tensorflow.python.ops.variables.Variable object at 0x10ab12dd0>
>>> K.eval(kvar)
array([[ 1.19591331,  0.68685907, -0.63814116],
   [ 0.92629528,  0.28055015,  1.70484698]], dtype=float32)
```

----

### count_params


```python
count_params(x)
```


Returns the number of scalars in a Keras variable.

__Arguments__

- __x__: Keras variable.

__Returns__

Integer, the number of scalars in `x`.

__Example__

```python
>>> kvar = K.zeros((2,3))
>>> K.count_params(kvar)
6
>>> K.eval(kvar)
array([[ 0.,  0.,  0.],
   [ 0.,  0.,  0.]], dtype=float32)
```

----

### cast


```python
cast(x, dtype)
```


Casts a tensor to a different dtype and returns it.

You can cast a Keras variable but it still returns a Keras tensor.

__Arguments__

- __x__: Keras tensor (or variable).
- __dtype__: String, either (`'float16'`, `'float32'`, or `'float64'`).

__Returns__

Keras tensor with dtype `dtype`.

__Example__

```python
>>> from keras import backend as K
>>> input = K.placeholder((2, 3), dtype='float32')
>>> input
<tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
__It doesn't work in-place as below.__

>>> K.cast(input, dtype='float16')
<tf.Tensor 'Cast_1:0' shape=(2, 3) dtype=float16>
>>> input
<tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
__you need to assign it.__

>>> input = K.cast(input, dtype='float16')
>>> input
<tf.Tensor 'Cast_2:0' shape=(2, 3) dtype=float16>
```

----

### update


```python
update(x, new_x)
```


Update the value of `x` to `new_x`.

__Arguments__

- __x__: A Variable.
- __new_x__: A tensor of same shape as `x`.

__Returns__

The variable `x` updated.

----

### update_add


```python
update_add(x, increment)
```


Update the value of `x` by adding `increment`.

__Arguments__

- __x__: A Variable.
- __increment__: A tensor of same shape as `x`.

__Returns__

The variable `x` updated.

----

### update_sub


```python
update_sub(x, decrement)
```


Update the value of `x` by subtracting `decrement`.

__Arguments__

- __x__: A Variable.
- __decrement__: A tensor of same shape as `x`.

__Returns__

The variable `x` updated.

----

### moving_average_update


```python
moving_average_update(x, value, momentum)
```


Compute the moving average of a variable.

__Arguments__

- __x__: A Variable.
- __value__: A tensor with the same shape as `variable`.
- __momentum__: The moving average momentum.

__Returns__

An Operation to update the variable.
----

### dot


```python
dot(x, y)
```


Multiplies 2 tensors (and/or variables) and returns a *tensor*.

When attempting to multiply a nD tensor
with a nD tensor, it reproduces the Theano behavior.
(e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

__Arguments__

- __x__: Tensor or variable.
- __y__: Tensor or variable.

__Returns__

A tensor, dot product of `x` and `y`.

__Examples__

```python
__dot product between tensors__

>>> x = K.placeholder(shape=(2, 3))
>>> y = K.placeholder(shape=(3, 4))
>>> xy = K.dot(x, y)
>>> xy
<tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
```

```python
__dot product between tensors__

>>> x = K.placeholder(shape=(32, 28, 3))
>>> y = K.placeholder(shape=(3, 4))
>>> xy = K.dot(x, y)
>>> xy
<tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
```

```python
__Theano-like behavior example__

>>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
>>> y = K.ones((4, 3, 5))
>>> xy = K.dot(x, y)
>>> K.int_shape(xy)
(2, 4, 5)
```

----

### batch_dot


```python
batch_dot(x, y, axes=None)
```


Batchwise dot product.

`batch_dot` is used to compute dot product of `x` and `y` when
`x` and `y` are data in batch, i.e. in a shape of
`(batch_size, :)`.
`batch_dot` results in a tensor or variable with less dimensions
than the input. If the number of dimensions is reduced to 1,
we use `expand_dims` to make sure that ndim is at least 2.

__Arguments__

- __x__: Keras tensor or variable with `ndim >= 2`.
- __y__: Keras tensor or variable with `ndim >= 2`.
- __axes__: list of (or single) int with target dimensions.
The lengths of `axes[0]` and `axes[1]` should be the same.

__Returns__

A tensor with shape equal to the concatenation of `x`'s shape
(less the dimension that was summed over) and `y`'s shape
(less the batch dimension and the dimension that was summed over).
If the final rank is 1, we reshape it to `(batch_size, 1)`.

__Examples__

Assume `x = [[1, 2], [3, 4]]` and `y = [[5, 6], [7, 8]]`
`batch_dot(x, y, axes=1) = [[17, 53]]` which is the main diagonal
of `x.dot(y.T)`, although we never have to calculate the off-diagonal
elements.

Shape inference:
Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
If `axes` is (1, 2), to find the output shape of resultant tensor,
loop through each dimension in `x`'s shape and `y`'s shape:

* `x.shape[0]` : 100 : append to output shape
* `x.shape[1]` : 20 : do not append to output shape,
dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
* `y.shape[0]` : 100 : do not append to output shape,
always ignore first dimension of `y`
* `y.shape[1]` : 30 : append to output shape
* `y.shape[2]` : 20 : do not append to output shape,
dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
`output_shape` = `(100, 30)`

```python
>>> x_batch = K.ones(shape=(32, 20, 1))
>>> y_batch = K.ones(shape=(32, 30, 20))
>>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
>>> K.int_shape(xy_batch_dot)
(32, 1, 30)
```

----

### transpose


```python
transpose(x)
```


Transposes a tensor and returns it.

__Arguments__

- __x__: Tensor or variable.

__Returns__

A tensor.

__Examples__

```python
>>> var = K.variable([[1, 2, 3], [4, 5, 6]])
>>> K.eval(var)
array([[ 1.,  2.,  3.],
   [ 4.,  5.,  6.]], dtype=float32)
>>> var_transposed = K.transpose(var)
>>> K.eval(var_transposed)
array([[ 1.,  4.],
   [ 2.,  5.],
   [ 3.,  6.]], dtype=float32)
```

```python
>>> input = K.placeholder((2, 3))
>>> input
<tf.Tensor 'Placeholder_11:0' shape=(2, 3) dtype=float32>
>>> input_transposed = K.transpose(input)
>>> input_transposed
<tf.Tensor 'transpose_4:0' shape=(3, 2) dtype=float32>

```

----

### gather


```python
gather(reference, indices)
```


Retrieves the elements of indices `indices` in the tensor `reference`.

__Arguments__

- __reference__: A tensor.
- __indices__: An integer tensor of indices.

__Returns__

A tensor of same type as `reference`.

----

### max


```python
max(x, axis=None, keepdims=False)
```


Maximum value in a tensor.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer, the axis to find maximum values.
- __keepdims__: A boolean, whether to keep the dimensions or not.
If `keepdims` is `False`, the rank of the tensor is reduced
by 1. If `keepdims` is `True`,
the reduced dimension is retained with length 1.

__Returns__

A tensor with maximum values of `x`.

----

### min


```python
min(x, axis=None, keepdims=False)
```


Minimum value in a tensor.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer, the axis to find minimum values.
- __keepdims__: A boolean, whether to keep the dimensions or not.
If `keepdims` is `False`, the rank of the tensor is reduced
by 1. If `keepdims` is `True`,
the reduced dimension is retained with length 1.

__Returns__

A tensor with miminum values of `x`.

----

### sum


```python
sum(x, axis=None, keepdims=False)
```


Sum of the values in a tensor, alongside the specified axis.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer, the axis to sum over.
- __keepdims__: A boolean, whether to keep the dimensions or not.
If `keepdims` is `False`, the rank of the tensor is reduced
by 1. If `keepdims` is `True`,
the reduced dimension is retained with length 1.

__Returns__

A tensor with sum of `x`.

----

### prod


```python
prod(x, axis=None, keepdims=False)
```


Multiplies the values in a tensor, alongside the specified axis.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer, the axis to compute the product.
- __keepdims__: A boolean, whether to keep the dimensions or not.
If `keepdims` is `False`, the rank of the tensor is reduced
by 1. If `keepdims` is `True`,
the reduced dimension is retained with length 1.

__Returns__

A tensor with the product of elements of `x`.

----

### cumsum


```python
cumsum(x, axis=0)
```


Cumulative sum of the values in a tensor, alongside the specified axis.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer, the axis to compute the sum.

__Returns__

A tensor of the cumulative sum of values of `x` along `axis`.

----

### cumprod


```python
cumprod(x, axis=0)
```


Cumulative product of the values in a tensor, alongside the specified axis.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer, the axis to compute the product.

__Returns__

A tensor of the cumulative product of values of `x` along `axis`.

----

### var


```python
var(x, axis=None, keepdims=False)
```


Variance of a tensor, alongside the specified axis.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer, the axis to compute the variance.
- __keepdims__: A boolean, whether to keep the dimensions or not.
If `keepdims` is `False`, the rank of the tensor is reduced
by 1. If `keepdims` is `True`,
the reduced dimension is retained with length 1.

__Returns__

A tensor with the variance of elements of `x`.

----

### std


```python
std(x, axis=None, keepdims=False)
```


Standard deviation of a tensor, alongside the specified axis.

__Arguments__

- __x__: A tensor or variable.
- __axis__: An integer, the axis to compute the standard deviation.
- __keepdims__: A boolean, whether to keep the dimensions or not.
If `keepdims` is `False`, the rank of the tensor is reduced
by 1. If `keepdims` is `True`,
the reduced dimension is retained with length 1.

__Returns__

A tensor with the standard deviation of elements of `x`.

----

### permute_dimensions


```python
permute_dimensions(x, pattern)
```


Permutes axes in a tensor.

__Arguments__

- __x__: Tensor or variable.
- __pattern__: A tuple of
dimension indices, e.g. `(0, 2, 1)`.

__Returns__

A tensor.

----

### resize_images


```python
resize_images(x, height_factor, width_factor, data_format)
```


Resizes the images contained in a 4D tensor.

__Arguments__

- __x__: Tensor or variable to resize.
- __height_factor__: Positive integer.
- __width_factor__: Positive integer.
- __data_format__: string, `"channels_last"` or `"channels_first"`.

__Returns__

A tensor.

__Raises__

- __ValueError__: if `data_format` is neither `"channels_last"` or `"channels_first"`.

----

### resize_volumes


```python
resize_volumes(x, depth_factor, height_factor, width_factor, data_format)
```


Resizes the volume contained in a 5D tensor.

__Arguments__

- __x__: Tensor or variable to resize.
- __depth_factor__: Positive integer.
- __height_factor__: Positive integer.
- __width_factor__: Positive integer.
- __data_format__: string, `"channels_last"` or `"channels_first"`.

__Returns__

A tensor.

__Raises__

- __ValueError__: if `data_format` is neither `"channels_last"` or `"channels_first"`.

----

### repeat_elements


```python
repeat_elements(x, rep, axis)
```


Repeats the elements of a tensor along an axis, like `np.repeat`.

If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
will have shape `(s1, s2 * rep, s3)`.

__Arguments__

- __x__: Tensor or variable.
- __rep__: Python integer, number of times to repeat.
- __axis__: Axis along which to repeat.

__Raises__

- __ValueError__: In case `x.shape[axis]` is undefined.

__Returns__

A tensor.

----

### repeat


```python
repeat(x, n)
```


Repeats a 2D tensor.

if `x` has shape (samples, dim) and `n` is `2`,
the output will have shape `(samples, 2, dim)`.

__Arguments__

- __x__: Tensor or variable.
- __n__: Python integer, number of times to repeat.

__Returns__

A tensor.

----

### arange


```python
arange(start, stop=None, step=1, dtype='int32')
```


Creates a 1D tensor containing a sequence of integers.

The function arguments use the same convention as
Theano's arange: if only one argument is provided,
it is in fact the "stop" argument.

The default type of the returned tensor is `'int32'` to
match TensorFlow's default.

__Arguments__

- __start__: Start value.
- __stop__: Stop value.
- __step__: Difference between two successive values.
- __dtype__: Integer dtype to use.

__Returns__

An integer tensor.


----

### tile


```python
tile(x, n)
```


Creates a tensor by tiling `x` by `n`.

__Arguments__

- __x__: A tensor or variable
- __n__: A list of integer. The length must be the same as the number of
dimensions in `x`.

__Returns__

A tiled tensor.

----

### flatten


```python
flatten(x)
```


Flatten a tensor.

__Arguments__

- __x__: A tensor or variable.

__Returns__

A tensor, reshaped into 1-D

----

### batch_flatten


```python
batch_flatten(x)
```


Turn a nD tensor into a 2D tensor with same 0th dimension.

In other words, it flattens each data samples of a batch.

__Arguments__

- __x__: A tensor or variable.

__Returns__

A tensor.

----

### expand_dims


```python
expand_dims(x, axis=-1)
```


Adds a 1-sized dimension at index "axis".

__Arguments__

- __x__: A tensor or variable.
- __axis__: Position where to add a new axis.

__Returns__

A tensor with expanded dimensions.

----

### squeeze


```python
squeeze(x, axis)
```


Removes a 1-dimension from the tensor at index "axis".

__Arguments__

- __x__: A tensor or variable.
- __axis__: Axis to drop.

__Returns__

A tensor with the same data as `x` but reduced dimensions.

----

### temporal_padding


```python
temporal_padding(x, padding=(1, 1))
```


Pads the middle dimension of a 3D tensor.

__Arguments__

- __x__: Tensor or variable.
- __padding__: Tuple of 2 integers, how many zeros to
add at the start and end of dim 1.

__Returns__

A padded 3D tensor.

----

### spatial_2d_padding


```python
spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None)
```


Pads the 2nd and 3rd dimensions of a 4D tensor.

__Arguments__

- __x__: Tensor or variable.
- __padding__: Tuple of 2 tuples, padding pattern.
- __data_format__: string, `"channels_last"` or `"channels_first"`.

__Returns__

A padded 4D tensor.

__Raises__

- __ValueError__: if `data_format` is neither `"channels_last"` or `"channels_first"`.

----

### spatial_3d_padding


```python
spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None)
```


Pads 5D tensor with zeros along the depth, height, width dimensions.

Pads these dimensions with respectively
"padding[0]", "padding[1]" and "padding[2]" zeros left and right.

For 'channels_last' data_format,
the 2nd, 3rd and 4th dimension will be padded.
For 'channels_first' data_format,
the 3rd, 4th and 5th dimension will be padded.

__Arguments__

- __x__: Tensor or variable.
- __padding__: Tuple of 3 tuples, padding pattern.
- __data_format__: string, `"channels_last"` or `"channels_first"`.

__Returns__

A padded 5D tensor.

__Raises__

- __ValueError__: if `data_format` is neither `"channels_last"` or `"channels_first"`.


----

### stack


```python
stack(x, axis=0)
```


Stacks a list of rank `R` tensors into a rank `R+1` tensor.

__Arguments__

- __x__: List of tensors.
- __axis__: Axis along which to perform stacking.

__Returns__

A tensor.

----

### one_hot


```python
one_hot(indices, num_classes)
```


Computes the one-hot representation of an integer tensor.

__Arguments__

- __indices__: nD integer tensor of shape
`(batch_size, dim1, dim2, ... dim(n-1))`
- __num_classes__: Integer, number of classes to consider.

__Returns__

(n + 1)D one hot representation of the input
with shape `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`

__Returns__

The one-hot tensor.

----

### reverse


```python
reverse(x, axes)
```


Reverse a tensor along the specified axes.

__Arguments__

- __x__: Tensor to reverse.
- __axes__: Integer or iterable of integers.
Axes to reverse.

__Returns__

A tensor.

----

### get_value


```python
get_value(x)
```


Returns the value of a variable.

__Arguments__

- __x__: input variable.

__Returns__

A Numpy array.

----

### batch_get_value


```python
batch_get_value(ops)
```


Returns the value of more than one tensor variable.

__Arguments__

- __ops__: list of ops to run.

__Returns__

A list of Numpy arrays.

----

### set_value


```python
set_value(x, value)
```


Sets the value of a variable, from a Numpy array.

__Arguments__

- __x__: Tensor to set to a new value.
- __value__: Value to set the tensor to, as a Numpy array
(of the same shape).

----

### batch_set_value


```python
batch_set_value(tuples)
```


Sets the values of many tensor variables at once.

__Arguments__

- __tuples__: a list of tuples `(tensor, value)`.
`value` should be a Numpy array.

----

### get_variable_shape


```python
get_variable_shape(x)
```


Returns the shape of a variable.

__Arguments__

- __x__: A variable.

__Returns__

A tuple of integers.

----

### print_tensor


```python
print_tensor(x, message='')
```


Prints `message` and the tensor value when evaluated.

__Arguments__

- __x__: Tensor to print.
- __message__: Message to print jointly with the tensor.

__Returns__

The same tensor `x`, unchanged.

----

### function


```python
function(inputs, outputs, updates=None)
```


Instantiates a Keras function.

__Arguments__

- __inputs__: List of placeholder tensors.
- __outputs__: List of output tensors.
- __updates__: List of update ops.
- __**kwargs__: Passed to `tf.Session.run`.

__Returns__

Output values as Numpy arrays.

__Raises__

- __ValueError__: if invalid kwargs are passed in.

----

### gradients


```python
gradients(loss, variables)
```


Returns the gradients of `variables` w.r.t. `loss`.

__Arguments__

- __loss__: Scalar tensor to minimize.
- __variables__: List of variables.

__Returns__

A gradients tensor.

----

### stop_gradient


```python
stop_gradient(variables)
```


Returns `variables` but with zero gradient w.r.t. every other variable.

__Arguments__

- __variables__: List of variables.

__Returns__

The same list of variables.

----

### rnn


```python
rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None)
```


Iterates over the time dimension of a tensor.

__Arguments__

- __step_function__: RNN step function.
- __Parameters__:
	- __input__: tensor with shape `(samples, ...)` (no time dimension),
	representing input for the batch of samples at a certain
	time step.
	- __states__: list of tensors.
- __Returns__:
	- __output__: tensor with shape `(samples, output_dim)`
	(no time dimension).
	- __new_states__: list of tensors, same length and shapes
	as 'states'. The first state in the list must be the
	output tensor at the previous timestep.
- __inputs__: tensor of temporal data of shape `(samples, time, ...)`
(at least 3D).
- __initial_states__: tensor with shape (samples, output_dim)
(no time dimension),
containing the initial values for the states used in
the step function.
- __go_backwards__: boolean. If True, do the iteration over the time
dimension in reverse order and return the reversed sequence.
- __mask__: binary tensor with shape `(samples, time, 1)`,
with a zero for every element that is masked.
- __constants__: a list of constant values passed at each step.
- __unroll__: whether to unroll the RNN or to use a symbolic loop (`while_loop` or `scan` depending on backend).
- __input_length__: not relevant in the TensorFlow implementation.
Must be specified if using unrolling with Theano.

__Returns__

A tuple, `(last_output, outputs, new_states)`.

- __last_output__: the latest output of the rnn, of shape `(samples, ...)`
- __outputs__: tensor with shape `(samples, time, ...)` where each
	entry `outputs[s, t]` is the output of the step function
	at time `t` for sample `s`.
- __new_states__: list of tensors, latest states returned by
	the step function, of shape `(samples, ...)`.

__Raises__

- __ValueError__: if input dimension is less than 3.
- __ValueError__: if `unroll` is `True` but input timestep is not a fixed number.
- __ValueError__: if `mask` is provided (not `None`) but states is not provided
(`len(states)` == 0).

----

### conv3d


```python
conv3d(x, kernel, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1))
```


3D convolution.

__Arguments__

- __x__: Tensor or variable.
- __kernel__: kernel tensor.
- __strides__: strides tuple.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
Whether to use Theano or TensorFlow data format
for inputs/kernels/outputs.
- __dilation_rate__: tuple of 3 integers.

__Returns__

A tensor, result of 3D convolution.

__Raises__

- __ValueError__: if `data_format` is neither `channels_last` or `channels_first`.

----

### pool2d


```python
pool2d(x, pool_size, strides=(1, 1), padding='valid', data_format=None, pool_mode='max')
```


2D Pooling.

__Arguments__

- __x__: Tensor or variable.
- __pool_size__: tuple of 2 integers.
- __strides__: tuple of 2 integers.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __pool_mode__: string, `"max"` or `"avg"`.

__Returns__

A tensor, result of 2D pooling.

__Raises__

- __ValueError__: if `data_format` is neither `"channels_last"` or `"channels_first"`.
- __ValueError__: if `pool_mode` is neither `"max"` or `"avg"`.

----

### pool3d


```python
pool3d(x, pool_size, strides=(1, 1, 1), padding='valid', data_format=None, pool_mode='max')
```


3D Pooling.

__Arguments__

- __x__: Tensor or variable.
- __pool_size__: tuple of 3 integers.
- __strides__: tuple of 3 integers.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __pool_mode__: string, `"max"` or `"avg"`.

__Returns__

A tensor, result of 3D pooling.

__Raises__

- __ValueError__: if `data_format` is neither `"channels_last"` or `"channels_first"`.
- __ValueError__: if `pool_mode` is neither `"max"` or `"avg"`.

----

### bias_add


```python
bias_add(x, bias, data_format=None)
```


Adds a bias vector to a tensor.

__Arguments__

- __x__: Tensor or variable.
- __bias__: Bias tensor to add.
- __data_format__: string, `"channels_last"` or `"channels_first"`.

__Returns__

Output tensor.

__Raises__

- __ValueError__: In one of the two cases below:
	1. invalid `data_format` argument.
	2. invalid bias shape.
	   the bias should be either a vector or
	   a tensor with ndim(x) - 1 dimension

----

### random_normal


```python
random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
```


Returns a tensor with normal distribution of values.

__Arguments__

- __shape__: A tuple of integers, the shape of tensor to create.
- __mean__: A float, mean of the normal distribution to draw samples.
- __stddev__: A float, standard deviation of the normal distribution
to draw samples.
- __dtype__: String, dtype of returned tensor.
- __seed__: Integer, random seed.

__Returns__

A tensor.

----

### random_uniform


```python
random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None)
```


Returns a tensor with uniform distribution of values.

__Arguments__

- __shape__: A tuple of integers, the shape of tensor to create.
- __minval__: A float, lower boundary of the uniform distribution
to draw samples.
- __maxval__: A float, upper boundary of the uniform distribution
to draw samples.
- __dtype__: String, dtype of returned tensor.
- __seed__: Integer, random seed.

__Returns__

A tensor.

----

### random_binomial


```python
random_binomial(shape, p=0.0, dtype=None, seed=None)
```


Returns a tensor with random binomial distribution of values.

__Arguments__

- __shape__: A tuple of integers, the shape of tensor to create.
- __p__: A float, `0. <= p <= 1`, probability of binomial distribution.
- __dtype__: String, dtype of returned tensor.
- __seed__: Integer, random seed.

__Returns__

A tensor.

----

### truncated_normal


```python
truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
```


Returns a tensor with truncated random normal distribution of values.

The generated values follow a normal distribution
with specified mean and standard deviation,
except that values whose magnitude is more than
two standard deviations from the mean are dropped and re-picked.

__Arguments__

- __shape__: A tuple of integers, the shape of tensor to create.
- __mean__: Mean of the values.
- __stddev__: Standard deviation of the values.
- __dtype__: String, dtype of returned tensor.
- __seed__: Integer, random seed.

__Returns__

A tensor.

----

### ctc_label_dense_to_sparse


```python
ctc_label_dense_to_sparse(labels, label_lengths)
```


Converts CTC labels from dense to sparse.

__Arguments__

- __labels__: dense CTC labels.
- __label_lengths__: length of the labels.

__Returns__

A sparse tensor representation of the lablels.

----

### ctc_batch_cost


```python
ctc_batch_cost(y_true, y_pred, input_length, label_length)
```


Runs CTC loss algorithm on each batch element.

__Arguments__

- __y_true__: tensor `(samples, max_string_length)`
containing the truth labels.
- __y_pred__: tensor `(samples, time_steps, num_categories)`
containing the prediction, or output of the softmax.
- __input_length__: tensor `(samples, 1)` containing the sequence length for
each batch item in `y_pred`.
- __label_length__: tensor `(samples, 1)` containing the sequence length for
each batch item in `y_true`.

__Returns__

Tensor with shape (samples,1) containing the
CTC loss of each element.

----

### ctc_decode


```python
ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1)
```


Decodes the output of a softmax.

Can use either greedy search (also known as best path)
or a constrained dictionary search.

__Arguments__

- __y_pred__: tensor `(samples, time_steps, num_categories)`
containing the prediction, or output of the softmax.
- __input_length__: tensor `(samples, )` containing the sequence length for
each batch item in `y_pred`.
- __greedy__: perform much faster best-path search if `true`.
This does not use a dictionary.
- __beam_width__: if `greedy` is `false`: a beam search decoder will be used
with a beam of this width.
- __top_paths__: if `greedy` is `false`,
how many of the most probable paths will be returned.

__Returns__

- __Tuple__:
- __List__: if `greedy` is `true`, returns a list of one element that
	contains the decoded sequence.
	If `false`, returns the `top_paths` most probable
	decoded sequences.
	- __Important__: blank labels are returned as `-1`.
Tensor `(top_paths, )` that contains
	the log probability of each decoded sequence.

----

### map_fn


```python
map_fn(fn, elems, name=None, dtype=None)
```


Map the function fn over the elements elems and return the outputs.

__Arguments__

- __fn__: Callable that will be called upon each element in elems
- __elems__: tensor
- __name__: A string name for the map node in the graph
- __dtype__: Output data type.

__Returns__

Tensor with dtype `dtype`.

----

### foldl


```python
foldl(fn, elems, initializer=None, name=None)
```


Reduce elems using fn to combine them from left to right.

__Arguments__

- __fn__: Callable that will be called upon each element in elems and an
accumulator, for instance `lambda acc, x: acc + x`
- __elems__: tensor
- __initializer__: The first value used (`elems[0]` in case of None)
- __name__: A string name for the foldl node in the graph

__Returns__

Tensor with same type and shape as `initializer`.

----

### foldr


```python
foldr(fn, elems, initializer=None, name=None)
```


Reduce elems using fn to combine them from right to left.

__Arguments__

- __fn__: Callable that will be called upon each element in elems and an
accumulator, for instance `lambda acc, x: acc + x`
- __elems__: tensor
- __initializer__: The first value used (`elems[-1]` in case of None)
- __name__: A string name for the foldr node in the graph

__Returns__

Same type and shape as initializer

----

### local_conv1d


```python
local_conv1d(inputs, kernel, kernel_size, strides, data_format=None)
```


Apply 1D conv with un-shared weights.

__Arguments__

- __inputs__: 3D tensor with shape: (batch_size, steps, input_dim)
- __kernel__: the unshared weight for convolution,
	with shape (output_length, feature_dim, filters)
- __kernel_size__: a tuple of a single integer,
	 specifying the length of the 1D convolution window
- __strides__: a tuple of a single integer,
	 specifying the stride length of the convolution
- __data_format__: the data format, channels_first or channels_last

__Returns__

the tensor after 1d conv with un-shared weights, with shape (batch_size, output_lenght, filters)

__Raises__

- __ValueError__: if `data_format` is neither `channels_last` or `channels_first`.

----

### local_conv2d


```python
local_conv2d(inputs, kernel, kernel_size, strides, output_shape, data_format=None)
```


Apply 2D conv with un-shared weights.

__Arguments__

- __inputs__: 4D tensor with shape:
	(batch_size, filters, new_rows, new_cols)
	if data_format='channels_first'
	or 4D tensor with shape:
	(batch_size, new_rows, new_cols, filters)
	if data_format='channels_last'.
- __kernel__: the unshared weight for convolution,
	with shape (output_items, feature_dim, filters)
- __kernel_size__: a tuple of 2 integers, specifying the
	 width and height of the 2D convolution window.
- __strides__: a tuple of 2 integers, specifying the strides
	 of the convolution along the width and height.
- __output_shape__: a tuple with (output_row, output_col)
- __data_format__: the data format, channels_first or channels_last

__Returns__

A 4d tensor with shape:
(batch_size, filters, new_rows, new_cols)
if data_format='channels_first'
or 4D tensor with shape:
(batch_size, new_rows, new_cols, filters)
if data_format='channels_last'.

__Raises__

- __ValueError__: if `data_format` is neither
	`channels_last` or `channels_first`.






