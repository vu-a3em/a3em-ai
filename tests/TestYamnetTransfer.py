from microesc.datasets.DirectoryDataSet import DirectoryDataSet
from microesc.classification.Yamnet import create_yamnet_model, YamnetParams, WaveformToLogMel
from microesc.detection.SpectralFluxDetector import SpectralFluxDetector
from microesc import keras
import tensorflow_model_optimization as tfmot
import microesc.tools as tools
import tensorflow as tf
import numpy as np

ignore_dirs = ['Gunshot',  'Fireworks', 'Drums', 'Engine', 'Noise', 'Whistle']

# Generate the Yamnet model
params = YamnetParams()
model = create_yamnet_model(params, load_pretrained_weights=True, freeze_pretrained_layers=True)
params.num_classes = 38 - len(ignore_dirs) 

# Remove the last layers and add new ones for transfer learning
for _ in range(1): #range(8 + 6):
  model.pop()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # type: ignore
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.summary()

classifier = keras.Sequential(name='Yamnet_Transfer_Classifier')
classifier.add(keras.Input(shape=model.output.shape[1:]))

# classifier.add(keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=2, depth_multiplier=1, padding=params.conv_padding, use_bias=False))
# classifier.add(keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon))
# classifier.add(keras.layers.ReLU(max_value=6.0))
# classifier.add(keras.layers.Conv2D(filters=1024, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False))
# classifier.add(keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon))
# classifier.add(keras.layers.ReLU(max_value=6.0))

# classifier.add(keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=1, depth_multiplier=1, padding=params.conv_padding, use_bias=False))
# classifier.add(keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon))
# classifier.add(keras.layers.ReLU(max_value=6.0))
# classifier.add(keras.layers.Conv2D(filters=1024, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False))
# classifier.add(keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon))
# classifier.add(keras.layers.ReLU(max_value=6.0))
# classifier.add(keras.layers.GlobalAveragePooling2D())

# classifier.add(keras.layers.Dense(128, activation='relu',
#                                   kernel_regularizer=keras.regularizers.l2(1e-4), use_bias=False))
# classifier.add(keras.layers.BatchNormalization(center=True, scale=True, epsilon=params.batchnorm_epsilon))
# classifier.add(keras.layers.Dropout(0.3))

classifier.add(keras.layers.Dense(params.num_classes, activation='softmax', use_bias=True))

initial_lr = 2e-4
decay_steps = 3000  # You may tune this value based on your dataset size/epochs
lr_schedule = keras.optimizers.schedules.CosineDecay(
  initial_learning_rate=initial_lr,
  decay_steps=decay_steps,
  alpha=1e-2  # Final learning rate as a fraction of initial_lr
)
classifier.compile(
  optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),  # type: ignore
  loss=keras.losses.SparseCategoricalCrossentropy(),
  metrics=[keras.metrics.SparseCategoricalAccuracy()]
)
classifier.summary()

# Create the full audio dataset and split it into a training and testing dataset
dataset_path = '/isis/home/steing/AIDataSet'
event_detector = SpectralFluxDetector(9.0, 8000, 512, 256, False, 150.0, 1800.0, 0.100)


dataset = DirectoryDataSet(dataset_path, params.sample_rate, params.patch_window_seconds + params.stft_window_seconds - params.stft_hop_seconds, 0.8, False, 0.3, event_detector, 0.1, ignore_directories=ignore_dirs)

batch_size = 32
train_ds = dataset.train_dataset(batch_size=batch_size)
test_ds = dataset.test_dataset(batch_size=batch_size)
dataset.summary()

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


# Train and save the best model
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history: keras.callbacks.History = classifier.fit(train_ds, epochs=10000, validation_data=test_ds, callbacks=[callback], verbose=2)  # type: ignore
classifier.evaluate(test_ds, return_dict=True)
tools.plot_training_history(history)
tools.plot_confusion_matrix(classifier, test_ds, dataset.idx_to_label)
classifier.save('yamnet_adapter.keras')

exit()

# Create a quantization-aware version of the trained model
from microesc.classification.Yamnet import WaveformToLogMel
def apply_quantization(layer: keras.layers.Layer):
  if not isinstance(layer, WaveformToLogMel):
    return tfmot.quantization.keras.quantize_annotate_layer(layer)
  return layer
quant_model = keras.models.clone_model(model, clone_function=apply_quantization)
with tfmot.quantization.keras.quantize_scope({'WaveformToLogMel': WaveformToLogMel}):
  quant_model: keras.Model = tfmot.quantization.keras.quantize_apply(quant_model)
quant_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # type: ignore
                    loss=keras.losses.SparseCategoricalCrossentropy(),
                    metrics=[keras.metrics.SparseCategoricalAccuracy()])
quant_model.summary()

# Carry out quantization-aware training for better quantized model accuracy
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = quant_model.fit(train_ds, epochs=10000, validation_data=test_ds, callbacks=[callback], verbose=2)  # type: ignore
quant_model.evaluate(test_ds, return_dict=True)
tools.plot_training_history(history)
tools.plot_confusion_matrix(quant_model, test_ds, dataset.idx_to_label)
quant_model.save('yamnet-quant-aware.keras')

# Convert the quantized model to quantized TFLite format
tools.convert_keras_to_tflite(quant_model, 'yamnet.tflite', True)
print(f"Quantized TFLite model accuracy: {tools.test_tflite_model('yamnet.tflite', test_ds)}")
