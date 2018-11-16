<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/noise.py#L10)</span>
### GaussianNoise

```python
keras.layers.noise.GaussianNoise(stddev)
```

Apply additive zero-centered Gaussian noise.

This is useful to mitigate overfitting
(you could see it as a form of random data augmentation).
Gaussian Noise (GS) is a natural choice as corruption process
for real valued inputs.

As it is a regularization layer, it is only active at training time.

__Arguments__

- __stddev__: float, standard deviation of the noise distribution.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as input.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/noise.py#L51)</span>
### GaussianDropout

```python
keras.layers.noise.GaussianDropout(rate)
```

Apply multiplicative 1-centered Gaussian noise.

As it is a regularization layer, it is only active at training time.

__Arguments__

- __rate__: float, drop probability (as with `Dropout`).
	The multiplicative noise will have
	standard deviation `sqrt(rate / (1 - rate))`.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as input.

__References__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
