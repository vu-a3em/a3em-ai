
import os
from microesc.datasets.DirectoryDataSet import DirectoryDataSet
from microesc.datasets.AugmentedDirectoryDataSet import AugmentedDirectoryDataSet
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
if 'SEED' in os.environ:
  seed = int(os.environ['SEED'])
else:
  seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# ignore_dirs = ['Gunshot',  'Fireworks', 'Drums', 'Engine', 'Noise']
# For now, ignoring classes with way too many or too few samples
ignore_dirs = ['Gunshot', 'Whistle']

background_classes = ['Noise', 'VehicleExhaust', 'Wind']

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

# Get all class names from the dataset directory, excluding ignored directories and background classes
all_classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
all_classes = [d for d in all_classes if d not in ignore_dirs and d not in background_classes]

# Pick N random classes for few-shot learning
N = 5
selected_classes = np.random.choice(all_classes, N, replace=False).tolist()
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
)

batch_size = 16
train_ds = dataset.train_dataset(batch_size=batch_size)
test_ds = dataset.test_dataset(batch_size=batch_size)
dataset.summary()

# Precompute and rebuild datasets
X_train, y_train = generate_embedding_dataset(train_ds)
X_test, y_test = generate_embedding_dataset(test_ds)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train) if len(y_train) > 0 else 1).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print(f"Extracted {X_train.shape[0]} training embeddings and {X_test.shape[0]} testing embeddings for few-shot classes.")

# Create new classifier model for the extracted features
classifier = keras.Sequential([
    keras.layers.Input(shape=model.output.shape[1:], dtype='float32'),
    keras.layers.Dense(units=256, use_bias=True, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=128, use_bias=True, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=len(selected_classes), use_bias=True, activation=params.classifier_activation)
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

history = classifier.fit(train_ds, epochs=10000, validation_data=test_ds, callbacks=[callback, lr_schedule], verbose=2)
val_eval = classifier.evaluate(test_ds, return_dict=True)

tools.plot_training_history(history, save_path=f'yamnetmini_training_history_fewshotclassifier_{seed}.png')
tools.plot_confusion_matrix(classifier, test_ds, dataset.idx_to_label, save_path=f'yamnetmini_confusion_matrix_fewshotclassifier_{seed}.png')
print(f"Yamnet-Mini + classifier achieved test accuracy {val_eval['sparse_categorical_accuracy']*100:.2f}%")

#classifier.save('yamnet_adapter.keras')
