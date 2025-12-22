import os
from microesc.datasets.DirectoryDataSet import DirectoryDataSet
from microesc.datasets.AugmentedDirectoryDataSet import AugmentedDirectoryDataSet
from microesc.classification.Yamnet import create_yamnet_model, YamnetParams, WaveformToLogMel
from microesc.detection.SpectralFluxDetector import SpectralFluxDetector
from microesc import keras
import tensorflow_model_optimization as tfmot
import microesc.tools as tools
from microesc.classification.training import TrainingConfig, generate_embedding_dataset, build_classifier_head
import itertools
import tensorflow as tf
import numpy as np

from microesc.classification.yamnetmini import build_mini_yamnet_model, DEFAULT_BLOCKS

#set seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ignore_dirs = ['Gunshot',  'Fireworks', 'Drums', 'Engine', 'Noise']
# For now, ignoring classes with way too many or too few samples
ignore_dirs = ['Gunshot', 'Whistle']

background_classes = ['Noise', 'VehicleExhaust', 'Wind']

# Generate the Yamnet model
params = YamnetParams()
params.num_classes = 38 - len(ignore_dirs) 

# Use default blocks from shared module
blocks = DEFAULT_BLOCKS
dense_units = []

evals = []

# Create the full audio dataset and split it into a training and testing dataset
dataset_path = '/isis/home/steing/AIDataSet'
event_detector = SpectralFluxDetector(9.0, 8000, 512, 256, False, 150.0, 1800.0, 0.100)

dataset = AugmentedDirectoryDataSet(
    dataset_path, 
    params.sample_rate,
    params.patch_window_seconds + params.stft_window_seconds - params.stft_hop_seconds, 
    0.8,
    False,
    0.3,
    event_detector,
    0.1,
    ignore_directories=ignore_dirs,
    background_classes=background_classes,
    background_to_event_ratio=[0.01, 0.3],
)

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

# Precompute and rebuild datasets using shared utility
X_train, y_train = generate_embedding_dataset(train_ds, model)
X_test, y_test = generate_embedding_dataset(test_ds, model)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train) if len(y_train) > 0 else 1).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Create new classifier model using shared utility
training_config = TrainingConfig(
    batch_size=batch_size,
    learning_rate=1e-4,
    epochs=10000,
    patience=10,
    dropout=0.25,
    hidden_units=[256, 128],
    activation='relu'
)

classifier = build_classifier_head(model.output.shape[1:], params.num_classes, training_config)

lr_schedule = training_config.create_lr_schedule('plateau', factor=0.5, schedule_patience=5, min_lr=1e-7)
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=training_config.patience, restore_best_weights=True)

classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=training_config.learning_rate),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
classifier.summary()

classifier.fit(train_ds, epochs=training_config.epochs, validation_data=test_ds, callbacks=[callback, lr_schedule], verbose=2)
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
