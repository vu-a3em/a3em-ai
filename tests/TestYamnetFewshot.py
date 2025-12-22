import os
from microesc.datasets.DirectoryDataSet import DirectoryDataSet
from microesc.datasets.AugmentedDirectoryDataSet import AugmentedDirectoryDataSet
from microesc.classification.Yamnet import create_yamnet_model, YamnetParams, WaveformToLogMel
from microesc.detection.SpectralFluxDetector import SpectralFluxDetector
from microesc import keras
import tensorflow_model_optimization as tfmot
import microesc.tools as tools
from microesc.tools.quantization import prepare_model_for_quantization
from microesc.classification.training import (
    TrainingConfig, generate_embedding_dataset, build_classifier_head,
    compute_none_bias_threshold, predict_with_none_bias
)
from microesc.datasets.utils import add_background_as_none
import itertools
import tensorflow as tf
import numpy as np

from microesc.classification.yamnetmini import build_mini_yamnet_model, DEFAULT_BLOCKS

#set seed for reproducibility
if 'SEED' in os.environ:
  seed = int(os.environ['SEED'])
else:
  seed = 42

tf.random.set_seed(seed)
np.random.seed(seed)

#seed = -1

# ignore_dirs = ['Gunshot',  'Fireworks', 'Drums', 'Engine', 'Noise']
# For now, ignoring classes with way too many or too few samples
ignore_dirs = ['Gunshot', 'Whistle']

background_classes = ['Noise', 'VehicleExhaust', 'Wind']

add_none_class = True

# Generate the Yamnet model
params = YamnetParams()
params.num_classes = 38 - len(ignore_dirs) 

# Use default blocks from shared module
blocks = DEFAULT_BLOCKS
dense_units = []

evals = []

# Create the full audio dataset and split it into a training and testing dataset
dataset_path = '/isis/home/steing/AIDataSet'
event_detector = SpectralFluxDetector(
  threshold=9.0, 
  sample_rate=8000,
  fft_length=512, hop_length=256, use_power_spectrum=False, min_frequency=150.0, max_frequency=1800.0, min_seconds_between_events=0.100)

# Get all class names from the dataset directory, excluding ignored directories and background classes
all_classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
all_classes = [d for d in all_classes if d not in ignore_dirs and d not in background_classes]

# Pick N random classes for few-shot learning
N = 5
selected_classes = np.random.choice(all_classes, N, replace=False).tolist()

selected_classes = [
  'ElephantRumble',
  'Rumination',
]

print(f"Selected classes for few-shot learning: {selected_classes}")

# Train without selected classes to simulate few-shot learning
ignore_dirs.extend([d for d in selected_classes])

if os.path.exists(f'yamnetmini-{seed}.keras'):
  print(f"Loading existing model yamnetmini-{seed}.keras")
  model = keras.models.load_model(f'yamnetmini-{seed}.keras', compile=False)
  model.summary()
else:
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
      use_metadata=True,
  )

  batch_size = 16
  train_ds = dataset.train_dataset(batch_size=batch_size)
  test_ds = dataset.test_dataset(batch_size=batch_size)
  dataset.summary()

  print(f"Training Yamnet-Mini with blocks={blocks} and dense_units={dense_units}")
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
  tools.plot_training_history(history, save_path=f'yamnetmini_training_history_{seed}.png')
  tools.plot_confusion_matrix(model, test_ds, dataset.idx_to_label, save_path=f'yamnetmini_confusion_matrix_{seed}.png')
  model.save(f'yamnetmini-{seed}.keras')

  print(f"Yamnet-Mini with blocks={blocks} and dense_units={dense_units} achieved test accuracy {val_eval['sparse_categorical_accuracy']*100:.2f}%")

# Pop off dense layer
model.pop()

print("Extracting embeddings for few-shot classes...")

# Make new datasets with only the selected few-shot classes
dataset = AugmentedDirectoryDataSet(
    dataset_path, 
    params.sample_rate,
    params.patch_window_seconds + params.stft_window_seconds - params.stft_hop_seconds, 
    0.2,
    False,
    0.3,
    event_detector,
    0.1,
    ignore_directories=[d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d not in selected_classes and d not in background_classes],
    background_classes=background_classes,
    background_to_event_ratio=[0.01, 0.3],
    max_samples_per_class=100,  # Limit to 100 samples per class for few-shot, will be up to 20 per class in training set (20-80 split)
    use_metadata=True,
)

if add_none_class:
  # Add background classes as "None" class

  # Compute max samples per background class to balance dataset
  max_samples = max(dataset.label_counts.values())
  #max_samples = 100
  per_bg_class = max_samples // len(background_classes)

  print(dataset.label_counts.values())
  #print(f"Max samples per background class: {per_bg_class}")

  for bg_class in background_classes:
    dataset.add_class_from_directory(
        label='None',
        path=os.path.join(dataset_path, bg_class),
        target_sample_rate_hz=params.sample_rate,
        target_clip_length=params.patch_window_seconds + params.stft_window_seconds - params.stft_hop_seconds,
        #max_samples=per_bg_class,
    )

batch_size = 16
train_ds = dataset.train_dataset(batch_size=batch_size)
test_ds = dataset.test_dataset(batch_size=batch_size)
dataset.summary()

train_ds.summary()
test_ds.summary()

# Precompute and rebuild datasets using shared utility
X_train, y_train = generate_embedding_dataset(train_ds, model)
X_test, y_test = generate_embedding_dataset(test_ds, model)

train_ds_orig, test_ds_orig = train_ds, test_ds
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train) if len(y_train) > 0 else 1).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print(f"Extracted {X_train.shape[0]} training embeddings and {X_test.shape[0]} testing embeddings for few-shot classes.")

# Create new classifier model using shared utility
training_config = TrainingConfig(
    batch_size=batch_size,
    learning_rate=1e-4,
    epochs=10000,
    patience=10,
    dropout=0.25,
    hidden_units=[256],
    activation='relu'
)

num_classes = (len(selected_classes) + 1) if add_none_class else len(selected_classes)
classifier = build_classifier_head(model.output.shape[1:], num_classes, training_config)

lr_schedule = training_config.create_lr_schedule('plateau', factor=0.5, schedule_patience=5, min_lr=1e-7)
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=training_config.patience, restore_best_weights=True)

classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=training_config.learning_rate),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
classifier.summary()

from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
cw = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

history = classifier.fit(train_ds, epochs=training_config.epochs, validation_data=test_ds, callbacks=[callback, lr_schedule], verbose=2, class_weight=class_weight_dict)
val_eval = classifier.evaluate(test_ds, return_dict=True)

tools.plot_training_history(history, save_path=f'yamnetmini_training_history_fewshotclassifier_{seed}.png')
tools.plot_confusion_matrix(classifier, test_ds, dataset.idx_to_label, save_path=f'yamnetmini_confusion_matrix_fewshotclassifier_{seed}.png')
print(f"Yamnet-Mini + classifier achieved test accuracy {val_eval['sparse_categorical_accuracy']*100:.2f}%")

from sklearn.metrics import classification_report, f1_score, roc_auc_score

y_pred = np.argmax(classifier.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=[dataset.idx_to_label[i] for i in range(len(dataset.idx_to_label))]))
print("Macro F1:", f1_score(y_test, y_pred, average='macro'))

# per-class AUROC (one-vs-rest)
probs = classifier.predict(X_test)
for i in range(probs.shape[1]):
    try:
        print(dataset.idx_to_label[i], roc_auc_score((y_test == i).astype(int), probs[:, i]))
    except Exception:
        pass

# Compute ROC-style curve and suggested threshold for 'None' biasing
if add_none_class:
  none_idx = dataset.label_to_idx['None']
  none_bias_threshold, none_bias_roc = compute_none_bias_threshold(classifier, test_ds, none_idx, target_fpr=0.10)
  print(f"None-bias ROC points (sample): {none_bias_roc[:5]}")

  # output plot of ROC curve
  import matplotlib.pyplot as plt
  fprs = [pt[2] for pt in none_bias_roc]
  tprs = [pt[1] for pt in none_bias_roc]
  plt.figure()
  plt.plot(fprs, tprs, label='None-bias ROC')
  plt.xlabel('False Positive Rate (None->Event)')
  plt.ylabel('True Positive Rate (Event->Event)')
  plt.title('ROC Curve for None-bias Thresholding')
  plt.grid()
  plt.savefig(f'yamnetmini_none_bias_roc_{seed}.png')

# Recombine feature extractor and classifier, by layers
for layer in classifier.layers:
  model.add(layer)

# Quick test of end-to-end model with original waveform inputs
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # type: ignore
                    loss=keras.losses.SparseCategoricalCrossentropy(),
                    metrics=[keras.metrics.SparseCategoricalAccuracy()])
test_eval = model.evaluate(test_ds_orig, return_dict=True)
print(f"End-to-end model test accuracy: {test_eval['sparse_categorical_accuracy']*100:.2f}%")

# All layers trainable for quantization-aware training
for layer in model.layers:
  if not isinstance(layer, WaveformToLogMel) and not isinstance(layer, keras.layers.GroupNormalization):
    layer.trainable = True

# Use shared quantization utilities
print("\n=== PRE-QUANTIZATION TEST ===")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
pre_quant_eval = model.evaluate(test_ds_orig, return_dict=True)
print(f"Pre-quantization test accuracy: {pre_quant_eval['sparse_categorical_accuracy']*100:.2f}%")

# Prepare model for quantization using shared utility
quant_model = prepare_model_for_quantization(model, verbose=True)
quant_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss=keras.losses.SparseCategoricalCrossentropy(),
                    metrics=[keras.metrics.SparseCategoricalAccuracy()])
quant_model.summary()

# Test initial quantized model accuracy
test_eval = quant_model.evaluate(test_ds_orig, return_dict=True)
print(f"Initial quantized model test accuracy: {test_eval['sparse_categorical_accuracy']*100:.2f}%")

# Carry out quantization-aware training for better quantized model accuracy
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = quant_model.fit(train_ds_orig, epochs=10000, validation_data=test_ds_orig, callbacks=[callback], verbose=2)
quant_model.evaluate(test_ds_orig, return_dict=True)
tools.plot_training_history(history)
tools.plot_confusion_matrix(quant_model, test_ds_orig, dataset.idx_to_label)
quant_model.save(f'yamnetmini-quant-aware-{seed}.keras')

# Convert the quantized model to quantized TFLite format
tools.convert_keras_to_tflite(quant_model, f'yamnetmini-quantized-{seed}.tflite', True)

print(f"Quantized TFLite model accuracy: {tools.test_tflite_model(f'yamnetmini-quantized-{seed}.tflite', test_ds_orig)}")