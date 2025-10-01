from microesc.datasets.DirectoryDataSet import DirectoryDataSet
from microesc.classification.Yamnet import create_yamnet_model, YamnetParams, WaveformToLogMel
from microesc.detection.SpectralFluxDetector import SpectralFluxDetector
from microesc import keras
import tensorflow_model_optimization as tfmot
import microesc.tools as tools
import itertools

from yamnetmini import build_mini_yamnet_model

ignore_dirs = ['Gunshot',  'Fireworks', 'Drums', 'Engine', 'Noise']

# Generate the Yamnet model
params = YamnetParams()
params.num_classes = 38 - len(ignore_dirs) 

block_configs = [
  # # Full
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
  # [
  #   (64, [3, 3], 1),
  #   (128, [3, 3], 2),
  #   (128, [3, 3], 1),
  #   (256, [3, 3], 2),
  #   (256, [3, 3], 1),
  # ],
  # Even smaller
  [
    (64, [3, 3], 1),
    (128, [3, 3], 2),
    (128, [3, 3], 1),
    (256, [3, 3], 2),
  ],
  # Tiny
  [
    (64, [3, 3], 1),
    (128, [3, 3], 2),
    (128, [3, 3], 1),
  ]
]

dense_configs = [
  [],
  [128],
  [128, 128],
]

evals = []

configs = list(itertools.product(block_configs, dense_configs))

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

for (blocks, dense_units) in configs:
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
  evals.append((blocks, dense_units, val_eval))
  #tools.plot_training_history(history)
  #tools.plot_confusion_matrix(model, test_ds, dataset.idx_to_label)
  #model.save('yamnetmini.keras')

for (blocks, dense_units, eval) in evals:
  print(f"Yamnet-Mini with blocks={blocks} and dense_units={dense_units} achieved test accuracy {eval['sparse_categorical_accuracy']*100:.2f}%")

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
