# The Sequential model API

To get started, read [this guide to the Keras Sequential model](/getting-started/sequential-model-guide).

## Useful attributes of Model

- `model.layers` is a list of the layers added to the model.


----

## Sequential model methods

### compile


```python
compile(self, optimizer, loss, metrics=None, sample_weight_mode=None)
```


Configures the learning process.

__Arguments__

- __optimizer__: str (name of optimizer) or optimizer object.
	See [optimizers](/optimizers).
- __loss__: str (name of objective function) or objective function.
	See [losses](/losses).
- __metrics__: list of metrics to be evaluated by the model
	during training and testing.
	Typically you will use `metrics=['accuracy']`.
	See [metrics](/metrics).
- __sample_weight_mode__: if you need to do timestep-wise
	sample weighting (2D weights), set this to "temporal".
	"None" defaults to sample-wise weights (1D).
- __**kwargs__: for Theano backend, these are passed into K.function.
	When using the Tensorflow backend, these are passed into
	`tf.Session.run`.

__Example__

```python
	model = Sequential()
	model.add(Dense(32, input_shape=(500,)))
	model.add(Dense(10, activation='softmax'))
	model.compile(optimizer='rmsprop',
		  loss='categorical_crossentropy',
		  metrics=['accuracy'])
```

----

### fit


```python
fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
```


Trains the model for a fixed number of epochs.

__Arguments__

- __x__: input data, as a Numpy array or list of Numpy arrays
	(if the model has multiple inputs).
- __y__: labels, as a Numpy array.
- __batch_size__: integer. Number of samples per gradient update.
- __epochs__: integer, the number of epochs to train the model.
- __verbose__: 0 for no logging to stdout,
	1 for progress bar logging, 2 for one log line per epoch.
- __callbacks__: list of `keras.callbacks.Callback` instances.
	List of callbacks to apply during training.
	See [callbacks](/callbacks).
- __validation_split__: float (0. < x < 1).
	Fraction of the data to use as held-out validation data.
- __validation_data__: tuple (x_val, y_val) or tuple
	(x_val, y_val, val_sample_weights) to be used as held-out
	validation data. Will override validation_split.
- __shuffle__: boolean or str (for 'batch').
	Whether to shuffle the samples at each epoch.
	'batch' is a special option for dealing with the
	limitations of HDF5 data; it shuffles in batch-sized chunks.
- __class_weight__: dictionary mapping classes to a weight value,
	used for scaling the loss function (during training only).
- __sample_weight__: Numpy array of weights for
	the training samples, used for scaling the loss function
	(during training only). You can either pass a flat (1D)
	Numpy array with the same length as the input samples
	(1:1 mapping between weights and samples),
	or in the case of temporal data,
	you can pass a 2D array with shape (samples, sequence_length),
	to apply a different weight to every timestep of every sample.
	In this case you should make sure to specify
	sample_weight_mode="temporal" in compile().
- __initial_epoch__: epoch at which to start training
	(useful for resuming a previous training run)

__Returns__

A `History` object. Its `History.history` attribute is
a record of training loss values and metrics values
at successive epochs, as well as validation loss values
and validation metrics values (if applicable).

__Raises__

- __RuntimeError__: if the model was never compiled.

----

### evaluate


```python
evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
```


Computes the loss on some input data, batch by batch.

__Arguments__

- __x__: input data, as a Numpy array or list of Numpy arrays
	(if the model has multiple inputs).
- __y__: labels, as a Numpy array.
- __batch_size__: integer. Number of samples per gradient update.
- __verbose__: verbosity mode, 0 or 1.
- __sample_weight__: sample weights, as a Numpy array.

__Returns__

Scalar test loss (if the model has no metrics)
or list of scalars (if the model computes other metrics).
The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.

__Raises__

- __RuntimeError__: if the model was never compiled.

----

### predict


```python
predict(self, x, batch_size=32, verbose=0)
```


Generates output predictions for the input samples.

The input samples are processed batch by batch.

__Arguments__

- __x__: the input data, as a Numpy array.
- __batch_size__: integer.
- __verbose__: verbosity mode, 0 or 1.

__Returns__

A Numpy array of predictions.

----

### train_on_batch


```python
train_on_batch(self, x, y, class_weight=None, sample_weight=None)
```


Single gradient update over one batch of samples.

__Arguments__

- __x__: input data, as a Numpy array or list of Numpy arrays
	(if the model has multiple inputs).
- __y__: labels, as a Numpy array.
- __class_weight__: dictionary mapping classes to a weight value,
	used for scaling the loss function (during training only).
- __sample_weight__: sample weights, as a Numpy array.

__Returns__

Scalar training loss (if the model has no metrics)
or list of scalars (if the model computes other metrics).
The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.

__Raises__

- __RuntimeError__: if the model was never compiled.

----

### test_on_batch


```python
test_on_batch(self, x, y, sample_weight=None)
```


Evaluates the model over a single batch of samples.

__Arguments__

- __x__: input data, as a Numpy array or list of Numpy arrays
	(if the model has multiple inputs).
- __y__: labels, as a Numpy array.
- __sample_weight__: sample weights, as a Numpy array.

__Returns__

Scalar test loss (if the model has no metrics)
or list of scalars (if the model computes other metrics).
The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.

__Raises__

- __RuntimeError__: if the model was never compiled.

----

### predict_on_batch


```python
predict_on_batch(self, x)
```


Returns predictions for a single batch of samples.

__Arguments__

- __x__: input data, as a Numpy array or list of Numpy arrays
	(if the model has multiple inputs).

__Returns__

A Numpy array of predictions.

----

### fit_generator


```python
fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)
```


Fits the model on data generated batch-by-batch by a Python generator.

The generator is run in parallel to the model, for efficiency.
For instance, this allows you to do real-time data augmentation
on images on CPU in parallel to training your model on GPU.

__Arguments__

- __generator__: A generator.
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
	be equal to the number of unique samples of your dataset
	divided by the batch size.
- __epochs__: Integer, total number of iterations on the data.
- __verbose__: Verbosity mode, 0, 1, or 2.
- __callbacks__: List of callbacks to be called during training.
- __validation_data__: This can be either
	- A generator for the validation data
	- A tuple (inputs, targets)
	- A tuple (inputs, targets, sample_weights).
- __validation_steps__: Only relevant if `validation_data`
	is a generator.
	Number of steps to yield from validation generator
	at the end of every epoch. It should typically
	be equal to the number of unique samples of your
	validation dataset divided by the batch size.
- __class_weight__: Dictionary mapping class indices to a weight
	for the class.
- __max_q_size__: Maximum size for the generator queue
- __workers__: Maximum number of processes to spin up
- __pickle_safe__: Ff True, use process based threading.
	Note that because
	this implementation relies on multiprocessing,
	you should not pass
	non picklable arguments to the generator
	as they can't be passed
	easily to children processes.
- __initial_epoch__: Epoch at which to start training
	(useful for resuming a previous training run)

__Returns__

A `History` object.

__Raises__

- __RuntimeError__: if the model was never compiled.

__Example__


```python
def generate_arrays_from_file(path):
	while 1:
	f = open(path)
	for line in f:
		# create Numpy arrays of input data
		# and labels, from each line in the file
		x, y = process_line(line)
		yield (x, y)
		f.close()

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
		steps_per_epoch=1000, epochs=10)
```

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
- __pickle_safe__: if True, use process based threading.
	Note that because this implementation
	relies on multiprocessing, you should not pass
	non picklable arguments to the generator
	as they can't be passed easily to children processes.

__Returns__

Scalar test loss (if the model has no metrics)
or list of scalars (if the model computes other metrics).
The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.

__Raises__

- __RuntimeError__: if the model was never compiled.

----

### predict_generator


```python
predict_generator(self, generator, steps, max_q_size=10, workers=1, pickle_safe=False, verbose=0)
```


Generates predictions for the input samples from a data generator.

The generator should return the same kind of data as accepted by
`predict_on_batch`.

__Arguments__

- __generator__: generator yielding batches of input samples.
- __steps__: Total number of steps (batches of samples)
	to yield from `generator` before stopping.
- __max_q_size__: maximum size for the generator queue
- __workers__: maximum number of processes to spin up
- __pickle_safe__: if True, use process based threading.
	Note that because this implementation
	relies on multiprocessing, you should not pass
	non picklable arguments to the generator
	as they can't be passed easily to children processes.
- __verbose__: verbosity mode, 0 or 1.

__Returns__

A Numpy array of predictions.

----

### get_layer


```python
get_layer(self, name=None, index=None)
```


Retrieve a layer that is part of the model.

Returns a layer based on either its name (unique)
or its index in the graph. Indices are based on
order of horizontal graph traversal (bottom-up).

__Arguments__

- __name__: string, name of layer.
- __index__: integer, index of layer.

__Returns__

A layer instance.
