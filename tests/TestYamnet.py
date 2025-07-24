from microesc.datasets.DirectoryDataSet import DirectoryDataSet
from microesc.classification.Yamnet import create_yamnet_model, YamnetParams
from microesc.detection.SpectralFluxDetector import SpectralFluxDetector
from microesc import keras
import tensorflow_model_optimization as tfmot
import microesc.tools as tools

# Generate the Yamnet model
params = YamnetParams()
params.num_classes = 38
model = create_yamnet_model(params, load_pretrained_weights=False, freeze_pretrained_layers=False)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # type: ignore
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.summary()

# Create the full audio dataset and split it into a training and testing dataset
dataset_path = '/Volumes/AIDataSets/DataSet'
event_detector = SpectralFluxDetector(9.0, 8000, 512, 256, False, 150.0, 1800.0, 0.100)
dataset = DirectoryDataSet(dataset_path, params.sample_rate, params.patch_window_seconds + params.stft_window_seconds - params.stft_hop_seconds, 0.8, False, 0.3, event_detector, 0.1)
train_ds = dataset.train_dataset(batch_size=32)
test_ds = dataset.test_dataset(batch_size=32)
dataset.summary()

# Train and save the best model
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history: keras.callbacks.History = model.fit(train_ds, epochs=10000, validation_data=test_ds, callbacks=[callback], verbose=2)  # type: ignore
model.evaluate(test_ds, return_dict=True)
tools.plot_training_history(history)
tools.plot_confusion_matrix(model, test_ds, dataset.idx_to_label)
model.save('yamnet.keras')

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
