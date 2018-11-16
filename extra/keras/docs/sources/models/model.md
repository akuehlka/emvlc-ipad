# Model class API

In the functional API, given some input tensor(s) and output tensor(s), you can instantiate a `Model` via:

```python
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)
```

This model will include all layers required in the computation of `b` given `a`.

In the case of multi-input or multi-output models, you can use lists as well:

```python
model = Model(inputs=[a1, a2], outputs=[b1, b3, b3])
```

For a detailed introduction of what `Model` can do, read [this guide to the Keras functional API](/getting-started/functional-api-guide).

## Useful attributes of Model

- `model.layers` is a flattened list of the layers comprising the model graph.
- `model.inputs` is the list of input tensors.
- `model.outputs` is the list of output tensors.

## Methods

### compile


```python
compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None)
```


Configures the model for training.

__Arguments__

- __optimizer__: str (name of optimizer) or optimizer object.
	See [optimizers](/optimizers).
- __loss__: str (name of objective function) or objective function.
	See [losses](/losses).
	If the model has multiple outputs, you can use a different loss
	on each output by passing a dictionary or a list of losses.
	The loss value that will be minimized by the model
	will then be the sum of all individual losses.
- __metrics__: list of metrics to be evaluated by the model
	during training and testing.
	Typically you will use `metrics=['accuracy']`.
	To specify different metrics for different outputs of a
	multi-output model, you could also pass a dictionary,
	such as `metrics={'output_a': 'accuracy'}`.
- __loss_weights__: Optional list or dictionary specifying scalar
	coefficients (Python floats) to weight the loss contributions
	of different model outputs.
	The loss value that will be minimized by the model
	will then be the *weighted sum* of all individual losses,
	weighted by the `loss_weights` coefficients.
	If a list, it is expected to have a 1:1 mapping
	to the model's outputs. If a tensor, it is expected to map
	output names (strings) to scalar coefficients.
- __sample_weight_mode__: if you need to do timestep-wise
	sample weighting (2D weights), set this to `"temporal"`.
	`None` defaults to sample-wise weights (1D).
	If the model has multiple outputs, you can use a different
	`sample_weight_mode` on each output by passing a
	dictionary or a list of modes.
- __**kwargs__: when using the Theano backend, these arguments
	are passed into K.function. When using the Tensorflow backend,
	these arguments are passed into `tf.Session.run`.

__Raises__

- __ValueError__: In case of invalid arguments for
	`optimizer`, `loss`, `metrics` or `sample_weight_mode`.

----

### fit


```python
fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
```


Trains the model for a fixed number of epochs (iterations on a dataset).

__Arguments__

- __x__: Numpy array of training data,
	or list of Numpy arrays if the model has multiple inputs.
	If all inputs in the model are named,
	you can also pass a dictionary
	mapping input names to Numpy arrays.
- __y__: Numpy array of target data,
	or list of Numpy arrays if the model has multiple outputs.
	If all outputs in the model are named,
	you can also pass a dictionary
	mapping output names to Numpy arrays.
- __batch_size__: integer. Number of samples per gradient update.
- __epochs__: integer, the number of times to iterate
	over the training data arrays.
- __verbose__: 0, 1, or 2. Verbosity mode.
	0 = silent, 1 = verbose, 2 = one log line per epoch.
- __callbacks__: list of callbacks to be called during training.
	See [callbacks](/callbacks).
- __validation_split__: float between 0 and 1:
	fraction of the training data to be used as validation data.
	The model will set apart this fraction of the training data,
	will not train on it, and will evaluate
	the loss and any model metrics
	on this data at the end of each epoch.
- __validation_data__: data on which to evaluate
	the loss and any model metrics
	at the end of each epoch. The model will not
	be trained on this data.
	This could be a tuple (x_val, y_val)
	or a tuple (x_val, y_val, val_sample_weights).
- __shuffle__: boolean, whether to shuffle the training data
	before each epoch.
- __class_weight__: optional dictionary mapping
	class indices (integers) to
	a weight (float) to apply to the model's loss for the samples
	from this class during training.
	This can be useful to tell the model to "pay more attention" to
	samples from an under-represented class.
- __sample_weight__: optional array of the same length as x, containing
	weights to apply to the model's loss for each sample.
	In the case of temporal data, you can pass a 2D array
	with shape (samples, sequence_length),
	to apply a different weight to every timestep of every sample.
	In this case you should make sure to specify
	sample_weight_mode="temporal" in compile().
- __initial_epoch__: epoch at which to start training
	(useful for resuming a previous training run)

__Returns__

A `History` instance. Its `history` attribute contains
all information collected during training.

__Raises__

- __ValueError__: In case of mismatch between the provided input data
	and what the model expects.

----

### evaluate


```python
evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
```


Returns the loss value & metrics values for the model in test mode.

Computation is done in batches.

__Arguments__

- __x__: Numpy array of test data,
	or list of Numpy arrays if the model has multiple inputs.
	If all inputs in the model are named,
	you can also pass a dictionary
	mapping input names to Numpy arrays.
- __y__: Numpy array of target data,
	or list of Numpy arrays if the model has multiple outputs.
	If all outputs in the model are named,
	you can also pass a dictionary
	mapping output names to Numpy arrays.
- __batch_size__: integer. Number of samples per gradient update.
- __verbose__: verbosity mode, 0 or 1.
- __sample_weight__: Array of weights to weight the contribution
	of different samples to the loss and metrics.

__Returns__

Scalar test loss (if the model has a single output and no metrics)
or list of scalars (if the model has multiple outputs
and/or metrics). The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.

----

### predict


```python
predict(self, x, batch_size=32, verbose=0)
```


Generates output predictions for the input samples.

Computation is done in batches.

__Arguments__

- __x__: the input data, as a Numpy array
	(or list of Numpy arrays if the model has multiple outputs).
- __batch_size__: integer.
- __verbose__: verbosity mode, 0 or 1.

__Returns__

Numpy array(s) of predictions.

__Raises__

- __ValueError__: In case of mismatch between the provided
	input data and the model's expectations,
	or in case a stateful model receives a number of samples
	that is not a multiple of the batch size.

----

### train_on_batch


```python
train_on_batch(self, x, y, sample_weight=None, class_weight=None)
```


Runs a single gradient update on a single batch of data.

__Arguments__

- __x__: Numpy array of training data,
	or list of Numpy arrays if the model has multiple inputs.
	If all inputs in the model are named,
	you can also pass a dictionary
	mapping input names to Numpy arrays.
- __y__: Numpy array of target data,
	or list of Numpy arrays if the model has multiple outputs.
	If all outputs in the model are named,
	you can also pass a dictionary
	mapping output names to Numpy arrays.
- __sample_weight__: optional array of the same length as x, containing
	weights to apply to the model's loss for each sample.
	In the case of temporal data, you can pass a 2D array
	with shape (samples, sequence_length),
	to apply a different weight to every timestep of every sample.
	In this case you should make sure to specify
	sample_weight_mode="temporal" in compile().
- __class_weight__: optional dictionary mapping
	class indices (integers) to
	a weight (float) to apply to the model's loss for the samples
	from this class during training.
	This can be useful to tell the model to "pay more attention" to
	samples from an under-represented class.

__Returns__

Scalar training loss
(if the model has a single output and no metrics)
or list of scalars (if the model has multiple outputs
and/or metrics). The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.

----

### test_on_batch


```python
test_on_batch(self, x, y, sample_weight=None)
```


Test the model on a single batch of samples.

__Arguments__

- __x__: Numpy array of test data,
	or list of Numpy arrays if the model has multiple inputs.
	If all inputs in the model are named,
	you can also pass a dictionary
	mapping input names to Numpy arrays.
- __y__: Numpy array of target data,
	or list of Numpy arrays if the model has multiple outputs.
	If all outputs in the model are named,
	you can also pass a dictionary
	mapping output names to Numpy arrays.
- __sample_weight__: optional array of the same length as x, containing
	weights to apply to the model's loss for each sample.
	In the case of temporal data, you can pass a 2D array
	with shape (samples, sequence_length),
	to apply a different weight to every timestep of every sample.
	In this case you should make sure to specify
	sample_weight_mode="temporal" in compile().

__Returns__

Scalar test loss (if the model has a single output and no metrics)
or list of scalars (if the model has multiple outputs
and/or metrics). The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.

----

### predict_on_batch


```python
predict_on_batch(self, x)
```


Returns predictions for a single batch of samples.

__Arguments__

- __x__: Input samples, as a Numpy array.

__Returns__

Numpy array(s) of predictions.

----

### fit_generator


```python
fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)
```


Fits the model on data yielded batch-by-batch by a Python generator.

The generator is run in parallel to the model, for efficiency.
For instance, this allows you to do real-time data augmentation
on images on CPU in parallel to training your model on GPU.

__Arguments__

- __generator__: a generator.
	The output of the generator must be either
	- a tuple (inputs, targets)
	- a tuple (inputs, targets, sample_weights).
	All arrays should contain the same number of samples.
	The generator is expected to loop over its data
	indefinitely. An epoch finishes when `steps_per_epoch`
	batches have been seen by the model.
- __steps_per_epoch__: Total number of steps (batches of samples)
	to yield from `generator` before declaring one epoch
	finished and starting the next epoch. It should typically
	be equal to the number of unique samples if your dataset
	divided by the batch size.
- __epochs__: integer, total number of iterations on the data.
- __verbose__: verbosity mode, 0, 1, or 2.
- __callbacks__: list of callbacks to be called during training.
- __validation_data__: this can be either
	- a generator for the validation data
	- a tuple (inputs, targets)
	- a tuple (inputs, targets, sample_weights).
- __validation_steps__: Only relevant if `validation_data`
	is a generator. Total number of steps (batches of samples)
	to yield from `generator` before stopping.
- __class_weight__: dictionary mapping class indices to a weight
	for the class.
- __max_q_size__: maximum size for the generator queue
- __workers__: maximum number of processes to spin up
	when using process based threading
- __pickle_safe__: if True, use process based threading.
	Note that because
	this implementation relies on multiprocessing,
	you should not pass
	non picklable arguments to the generator
	as they can't be passed
	easily to children processes.
- __initial_epoch__: epoch at which to start training
	(useful for resuming a previous training run)

__Returns__

A `History` object.

__Example__


```python
def generate_arrays_from_file(path):
	while 1:
	f = open(path)
	for line in f:
		# create numpy arrays of input data
		# and labels, from each line in the file
		x1, x2, y = process_line(line)
		yield ({'input_1': x1, 'input_2': x2}, {'output': y})
	f.close()

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
		steps_per_epoch=10000, epochs=10)
```

__Raises__

- __ValueError__: In case the generator yields
	data in an invalid format.

----

### evaluate_generator


```python
evaluate_generator(self, generator, steps, max_q_size=10, workers=1, pickle_safe=False)
```


Evaluates the model on a data generator.

The generator should return the same kind of data
as accepted by `test_on_batch`.

__Arguments__

- __generator__: Generator yielding tuples (inputs, targets)
	or (inputs, targets, sample_weights)
- __steps__: Total number of steps (batches of samples)
	to yield from `generator` before stopping.
- __max_q_size__: maximum size for the generator queue
- __workers__: maximum number of processes to spin up
	when using process based threading
- __pickle_safe__: if True, use process based threading.
	Note that because
	this implementation relies on multiprocessing,
	you should not pass
	non picklable arguments to the generator
	as they can't be passed
	easily to children processes.

__Returns__

Scalar test loss (if the model has a single output and no metrics)
or list of scalars (if the model has multiple outputs
and/or metrics). The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.

__Raises__

- __ValueError__: In case the generator yields
	data in an invalid format.

----

### predict_generator


```python
predict_generator(self, generator, steps, max_q_size=10, workers=1, pickle_safe=False, verbose=0)
```


Generates predictions for the input samples from a data generator.

The generator should return the same kind of data as accepted by
`predict_on_batch`.

__Arguments__

- __generator__: Generator yielding batches of input samples.
- __steps__: Total number of steps (batches of samples)
	to yield from `generator` before stopping.
- __max_q_size__: Maximum size for the generator queue.
- __workers__: Maximum number of processes to spin up
	when using process based threading
- __pickle_safe__: If `True`, use process based threading.
	Note that because
	this implementation relies on multiprocessing,
	you should not pass
	non picklable arguments to the generator
	as they can't be passed
	easily to children processes.
- __verbose__: verbosity mode, 0 or 1.

__Returns__

Numpy array(s) of predictions.

__Raises__

- __ValueError__: In case the generator yields
	data in an invalid format.

----

### get_layer


```python
get_layer(self, name=None, index=None)
```


Retrieves a layer based on either its name (unique) or index.

Indices are based on order of horizontal graph traversal (bottom-up).

__Arguments__

- __name__: String, name of layer.
- __index__: Integer, index of layer.

__Returns__

A layer instance.

__Raises__

- __ValueError__: In case of invalid layer name or index.

