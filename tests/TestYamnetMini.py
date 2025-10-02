import os
from microesc.datasets.DirectoryDataSet import DirectoryDataSet
from microesc.classification.Yamnet import create_yamnet_model, YamnetParams, WaveformToLogMel
from microesc.detection.SpectralFluxDetector import SpectralFluxDetector
from microesc import keras
import tensorflow_model_optimization as tfmot
import microesc.tools as tools
import itertools
import tensorflow as tf
import numpy as np

from microesc.classification.yamnetmini import build_mini_yamnet_model

#set seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ignore_dirs = ['Gunshot',  'Fireworks', 'Drums', 'Engine', 'Noise']
ignore_dirs = ['Gunshot']

# Generate the Yamnet model
params = YamnetParams()
params.num_classes = 38 - len(ignore_dirs) 

blocks = [
  # Full
  # [
  #   (64, [3, 3], 1),
  #   (128, [3, 3], 2),
  #   (128, [3, 3], 1),
  #   (256, [3, 3], 2),
  #   (256, [3, 3], 1),
  #   (512, [3, 3], 2),
  #   (512, [3, 3], 1),
  #   (512, [3, 3], 1),
  #   (512, [3, 3], 1),
  #   (512, [3, 3], 1),
  #   (512, [3, 3], 1),
  #   (1024, [3, 3], 2),
  #   (1024, [3, 3], 1),
  # ],
  # # Larger
  # [
  #   (64, [3, 3], 1),
  #   (128, [3, 3], 2),
  #   (128, [3, 3], 1),
  #   (256, [3, 3], 2),
  #   (256, [3, 3], 1),
  #   (512, [3, 3], 2),
  #   (512, [3, 3], 1),
  #   (1024, [3, 3], 2),
  # ],
  # # Original
  # [
  #   (64, [3, 3], 1),
  #   (128, [3, 3], 2),
  #   (128, [3, 3], 1),
  #   (256, [3, 3], 2),
  #   (256, [3, 3], 1),
  #   (512, [3, 3], 2),
  # ],
  # # Smaller
    (64, [3, 3], 1),
    (128, [3, 3], 2),
    (128, [3, 3], 1),
    (256, [3, 3], 2),
    (256, [3, 3], 1),
  # Even smaller
  # [
  #   (64, [3, 3], 1),
  #   (128, [3, 3], 2),
  #   (128, [3, 3], 1),
  #   (256, [3, 3], 2),
  # ]
  # Tiny
  # [
  #   (64, [3, 3], 1),
  #   (128, [3, 3], 2),
  #   (128, [3, 3], 1),
  # ]
]

dense_units = []

evals = []

# Create the full audio dataset and split it into a training and testing dataset
dataset_path = '/isis/home/steing/AIDataSet'
event_detector = SpectralFluxDetector(9.0, 8000, 512, 256, False, 150.0, 1800.0, 0.100)

dataset = DirectoryDataSet(dataset_path, params.sample_rate, params.patch_window_seconds + params.stft_window_seconds - params.stft_hop_seconds, 0.8, False, 0.3, event_detector, 0.1, ignore_directories=ignore_dirs)

batch_size = 16
train_ds = dataset.train_dataset(batch_size=batch_size)
test_ds = dataset.test_dataset(batch_size=batch_size)
dataset.summary()

#  Test for out-of-range values in the input data
# for x_ds, y_ds in train_ds:
#     print(x_ds.shape, y_ds.shape)

#     for x in x_ds:
#       for i in range(x.shape[0]):
#           if x[i] > 1.0 or x[i] < -1.0:
#             print(f"{i:03d}: {x[i]}")

if os.path.exists('yamnetmini.keras'):
  print("Loading existing Yamnet-Mini model from yamnetmini.keras")
  model = keras.models.load_model('yamnetmini.keras')
else:
  print(f"Testing Yamnet-Mini with blocks={blocks} and dense_units={dense_units}")
  model = build_mini_yamnet_model(params, blocks=blocks, dense_units=dense_units)
  # Learning rate schedule: reduce LR on plateau
  lr_schedule = keras.callbacks.ReduceLROnPlateau(
      monitor='val_loss',
      factor=0.5,
      patience=5,
      min_lr=1e-7,
      verbose=1
  )
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),  
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=[keras.metrics.SparseCategoricalAccuracy()])
  model.summary()


  # Train and save the best model
  callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
  history: keras.callbacks.History = model.fit(
      train_ds,
      epochs=10000,
      validation_data=test_ds,
      callbacks=[callback, lr_schedule],
      verbose=2
  )  # type: ignore
  val_eval = model.evaluate(test_ds, return_dict=True)
  tools.plot_training_history(history)
  tools.plot_confusion_matrix(model, test_ds, dataset.idx_to_label)
  model.save('yamnetmini.keras')

  print(f"Yamnet-Mini with blocks={blocks} and dense_units={dense_units} achieved test accuracy {val_eval['sparse_categorical_accuracy']*100:.2f}%")


# Pop off dense layer
model.pop()

# Use model for feature extraction on entire dataset
# Preprocess the datasets to extract embeddings using the frozen Yamnet model
def extract_yamnet_embeddings(waveforms, labels):
  embeddings = model(waveforms, training=False)
  return embeddings, labels

# Generate embeddings to numpy arrays once, then create new tf.data.Dataset objects
def generate_embedding_dataset(seq):
  """Takes a KerasDataSet (keras.utils.Sequence) and returns (X, y) numpy arrays of embeddings and labels."""
  embs = []
  lbls = []
  for i in range(len(seq)):
    batch_waveforms, batch_labels = seq[i]
    batch_embs = model(batch_waveforms, training=False)
    # Convert to numpy if it's a Tensor
    if hasattr(batch_embs, 'numpy'):
      batch_embs = batch_embs.numpy()
    embs.append(batch_embs)
    lbls.append(batch_labels)
  X = np.concatenate(embs, axis=0) if len(embs) > 0 else np.empty((0,) + model.output.shape[1:], dtype=np.float32)
  y = np.concatenate(lbls, axis=0) if len(lbls) > 0 else np.empty((0,), dtype=np.int64)
  return X, y

# Precompute and rebuild datasets
X_train, y_train = generate_embedding_dataset(train_ds)
X_test, y_test = generate_embedding_dataset(test_ds)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train) if len(y_train) > 0 else 1).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Create new classifier model for the extracted features
classifier = keras.Sequential([
    keras.layers.Input(shape=model.output.shape[1:], dtype='float32'),
    keras.layers.Dense(units=256, use_bias=True, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=128, use_bias=True, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=params.num_classes, use_bias=True, activation=params.classifier_activation)
])
lr_schedule = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
classifier.summary()

classifier.fit(train_ds, epochs=10000, validation_data=test_ds, callbacks=[callback, lr_schedule], verbose=2)
val_eval = classifier.evaluate(test_ds, return_dict=True)
print(f"Yamnet-Mini + classifier achieved test accuracy {val_eval['sparse_categorical_accuracy']*100:.2f}%")

#tools.plot_training_history(history)
#tools.plot_confusion_matrix(classifier, test_ds, dataset.idx_to_label)
classifier.save('yamnet_adapter.keras')


# Skip quantization-aware training for now
exit()

# Create a quantization-aware version of the trained model
# from microesc.classification.Yamnet import WaveformToLogMel
# def apply_quantization(layer: keras.layers.Layer):
#   if not isinstance(layer, WaveformToLogMel):
#     return tfmot.quantization.keras.quantize_annotate_layer(layer)
#   return layer
# quant_model = keras.models.clone_model(model, clone_function=apply_quantization)
# with tfmot.quantization.keras.quantize_scope({'WaveformToLogMel': WaveformToLogMel}):
#   quant_model: keras.Model = tfmot.quantization.keras.quantize_apply(quant_model)
# quant_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # type: ignore
#                     loss=keras.losses.SparseCategoricalCrossentropy(),
#                     metrics=[keras.metrics.SparseCategoricalAccuracy()])
# quant_model.summary()

# # Carry out quantization-aware training for better quantized model accuracy
# callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# history = quant_model.fit(train_ds, epochs=10000, validation_data=test_ds, callbacks=[callback], verbose=2)  # type: ignore
# quant_model.evaluate(test_ds, return_dict=True)
# tools.plot_training_history(history)
# tools.plot_confusion_matrix(quant_model, test_ds, dataset.idx_to_label)
# quant_model.save('yamnetmini-quant-aware.keras')

# # Convert the quantized model to quantized TFLite format
# tools.convert_keras_to_tflite(quant_model, 'yamnetmini.tflite', True)
# print(f"Quantized TFLite model accuracy: {tools.test_tflite_model('yamnetmini.tflite', test_ds)}")
