from microesc.datasets.DirectoryDataSet import DirectoryDataSet
from microesc.classification.Yamnet import create_yamnet_model, YamnetParams, WaveformToLogMel
from microesc.detection.SpectralFluxDetector import SpectralFluxDetector
from microesc import keras
import tensorflow_model_optimization as tfmot
import microesc.tools as tools
from microesc.classification.training import TrainingConfig, generate_embedding_dataset, build_classifier_head
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

# Build classifier using shared utility
training_config = TrainingConfig(
    batch_size=32,
    learning_rate=2e-4,
    epochs=10000,
    patience=10,
    dropout=0.0,
    hidden_units=[],  # Single dense layer only
    activation='relu'
)

# Create cosine decay LR schedule
lr_schedule = training_config.create_lr_schedule('cosine', decay_steps=3000, alpha=1e-2)

classifier = build_classifier_head(model.output.shape[1:], params.num_classes, training_config)

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

# Precompute and rebuild datasets using shared utility
X_train, y_train = generate_embedding_dataset(train_ds, model)
X_test, y_test = generate_embedding_dataset(test_ds, model)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train) if len(y_train) > 0 else 1).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Train and save the best model
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=training_config.patience, restore_best_weights=True)
history: keras.callbacks.History = classifier.fit(train_ds, epochs=training_config.epochs, validation_data=test_ds, callbacks=[callback], verbose=2)  # type: ignore
classifier.evaluate(test_ds, return_dict=True)
tools.plot_training_history(history)
tools.plot_confusion_matrix(classifier, test_ds, dataset.idx_to_label)
classifier.save('yamnet_adapter.keras')

exit()

# Create a quantization-aware version of the trained model
from microesc.classification.Yamnet import WaveformToLogMel
def apply_quantization(layer: keras.layers.Layer):
  if not isinstance(layer, WaveformToLogMel) and not isinstance(layer, keras.layers.GroupNormalization):
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
