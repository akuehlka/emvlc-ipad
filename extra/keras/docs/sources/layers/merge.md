<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/merge.py#L184)</span>
### Add

```python
keras.layers.merge.Add()
```

Layer that adds a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/merge.py#L199)</span>
### Multiply

```python
keras.layers.merge.Multiply()
```

Layer that multiplies (element-wise) a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/merge.py#L214)</span>
### Average

```python
keras.layers.merge.Average()
```

Layer that averages a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/merge.py#L229)</span>
### Maximum

```python
keras.layers.merge.Maximum()
```

Layer that computes the maximum (element-wise) a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/merge.py#L244)</span>
### Concatenate

```python
keras.layers.merge.Concatenate(axis=-1)
```

Layer that concatenates a list of inputs.

It takes as input a list of tensors,
all of the same shape expect for the concatenation axis,
and returns a single tensor, the concatenation of all inputs.

__Arguments__

- __axis__: Axis along which to concatenate.
- __**kwargs__: standard layer keyword arguments.

----

<span style="float:right;">[[source]](https://github.com/fchollet/keras/blob/master/keras/layers/merge.py#L335)</span>
### Dot

```python
keras.layers.merge.Dot(axes, normalize=False)
```

Layer that computes a dot product between samples in two tensors.

E.g. if applied to two tensors `a` and `b` of shape `(batch_size, n)`,
the output will be a tensor of shape `(batch_size, 1)`
where each entry `i` will be the dot product between
`a[i]` and `b[i]`.

__Arguments__

- __axes__: Integer or tuple of integers,
	axis or axes along which to take the dot product.
- __normalize__: Whether to L2-normalize samples along the
	dot product axis before taking the dot product.
	If set to True, then the output of the dot product
	is the cosine proximity between the two samples.
- __**kwargs__: Standard layer keyword arguments.

----

### add


```python
add(inputs)
```


Functional interface to the `Add` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the sum of the inputs.

----

### multiply


```python
multiply(inputs)
```


Functional interface to the `Multiply` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the element-wise product of the inputs.

----

### average


```python
average(inputs)
```


Functional interface to the `Average` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the average of the inputs.

----

### maximum


```python
maximum(inputs)
```


Functional interface to the `Maximum` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the element-wise maximum of the inputs.

----

### concatenate


```python
concatenate(inputs, axis=-1)
```


Functional interface to the `Concatenate` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __axis__: Concatenation axis.
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the concatenation of the inputs alongside axis `axis`.

----

### dot


```python
dot(inputs, axes, normalize=False)
```


Functional interface to the `Dot` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __axes__: Integer or tuple of integers,
axis or axes along which to take the dot product.
- __normalize__: Whether to L2-normalize samples along the
dot product axis before taking the dot product.
If set to True, then the output of the dot product
is the cosine proximity between the two samples.
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the dot product of the samples from the inputs.
