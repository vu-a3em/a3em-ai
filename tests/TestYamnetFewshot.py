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

seed = -1

# ignore_dirs = ['Gunshot',  'Fireworks', 'Drums', 'Engine', 'Noise']
# For now, ignoring classes with way too many or too few samples
ignore_dirs = ['Gunshot', 'Whistle']

background_classes = ['Noise', 'VehicleExhaust', 'Wind']

add_none_class = True

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

# Precompute and rebuild datasets
X_train, y_train = generate_embedding_dataset(train_ds)
X_test, y_test = generate_embedding_dataset(test_ds)

train_ds_orig, test_ds_orig = train_ds, test_ds
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train) if len(y_train) > 0 else 1).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print(f"Extracted {X_train.shape[0]} training embeddings and {X_test.shape[0]} testing embeddings for few-shot classes.")

# Create new classifier model for the extracted features
classifier = keras.Sequential([
    keras.layers.Input(shape=model.output.shape[1:], dtype='float32'),
    keras.layers.Dense(units=256, use_bias=True, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=(len(selected_classes) + 1) if add_none_class else len(selected_classes), use_bias=True, activation=params.classifier_activation)
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

from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
cw = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

history = classifier.fit(train_ds, epochs=10000, validation_data=test_ds, callbacks=[callback, lr_schedule], verbose=2, class_weight=class_weight_dict)
val_eval = classifier.evaluate(test_ds, return_dict=True)

tools.plot_training_history(history, save_path=f'yamnetmini_training_history_fewshotclassifier_{seed}.png')
tools.plot_confusion_matrix(classifier, test_ds, dataset.idx_to_label, save_path=f'yamnetmini_confusion_matrix_fewshotclassifier_{seed}.png')
print(f"Yamnet-Mini + classifier achieved test accuracy {val_eval['sparse_categorical_accuracy']*100:.2f}%")

# Helper: tune decision threshold to bias away from 'None' class

def compute_none_bias_threshold(model, val_ds, none_idx: int, target_fpr: float = 0.10):
  """Compute a decision threshold on (best_non_none_prob - none_prob).

  - model: trained classifier (softmax probs output).
  - val_ds: tf.data.Dataset of (X, y) for validation.
  - none_idx: integer index of the None/background class.
  - target_fpr: max allowed None->event false positive rate.

  Returns: (best_threshold, roc_points) where roc_points is a list of (t, tpr, fpr).
  """
  y_true_list, y_prob_list = [], []
  for xb, yb in val_ds:
    probs = model(xb, training=False).numpy()
    y_prob_list.append(probs)
    y_true_list.append(yb.numpy())

  if not y_true_list:
    raise ValueError("Empty validation dataset passed to compute_none_bias_threshold")

  y_true = np.concatenate(y_true_list)
  y_prob = np.concatenate(y_prob_list)

  num_classes = y_prob.shape[1]
  non_none_mask = np.ones(num_classes, dtype=bool)
  non_none_mask[none_idx] = False

  best_non_none_idx = np.argmax(y_prob[:, non_none_mask], axis=1)
  best_non_none_prob = y_prob[:, non_none_mask][np.arange(len(y_prob)), best_non_none_idx]
  none_prob = y_prob[:, none_idx]
  score = best_non_none_prob - none_prob

  is_event_true = (y_true != none_idx)

  def event_vs_none_metrics(t: float):
    choose_event = (score >= t)
    global_non_none_indices = np.arange(num_classes)[non_none_mask]
    pred = np.full_like(y_true, fill_value=none_idx)
    pred[choose_event] = global_non_none_indices[best_non_none_idx[choose_event]]

    is_event_pred = (pred != none_idx)

    tp = np.sum(is_event_pred & is_event_true)
    fp = np.sum(is_event_pred & ~is_event_true)
    fn = np.sum(~is_event_pred & is_event_true)
    tn = np.sum(~is_event_pred & ~is_event_true)

    tpr = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    return tpr, fpr

  ts = np.linspace(-1.0, 1.0, 201)
  roc = []
  best_t, best_tpr = ts[0], -1.0
  for t in ts:
    tpr, fpr = event_vs_none_metrics(t)
    roc.append((t, tpr, fpr))
    if fpr <= target_fpr and tpr > best_tpr:
      best_tpr = tpr
      best_t = t

  print(f"Selected None-bias threshold={best_t:.4f} with TPR={best_tpr:.4f} at FPR <= {target_fpr}")
  return best_t, roc


def predict_with_none_bias(model, x, none_idx: int, threshold: float):
  """Apply biased decision rule at inference time.

  - model: trained classifier (softmax output).
  - x: input batch.
  - none_idx: index of None/background class.
  - threshold: chosen on (best_non_none_prob - none_prob).
  """
  probs = model(x, training=False).numpy()
  num_classes = probs.shape[1]

  non_none_mask = np.ones(num_classes, dtype=bool)
  non_none_mask[none_idx] = False

  best_non_none_idx = np.argmax(probs[:, non_none_mask], axis=1)
  best_non_none_prob = probs[:, non_none_mask][np.arange(len(probs)), best_non_none_idx]
  none_prob = probs[:, none_idx]
  score = best_non_none_prob - none_prob

  global_non_none_indices = np.arange(num_classes)[non_none_mask]
  pred = np.full((len(probs),), fill_value=none_idx, dtype=int)
  choose_event = (score >= threshold)
  pred[choose_event] = global_non_none_indices[best_non_none_idx[choose_event]]
  return pred


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

# Create a quantization-aware version of the trained model
from microesc.classification.Yamnet import WaveformToLogMel

# Test the model BEFORE quantization to confirm accuracy
print("\n=== PRE-QUANTIZATION TEST ===")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
pre_quant_eval = model.evaluate(test_ds_orig, return_dict=True)
print(f"Pre-quantization test accuracy: {pre_quant_eval['sparse_categorical_accuracy']*100:.2f}%")

# KEEP GroupNormalization layers - they are essential!
# Only quantize Dense, Conv2D, and DepthwiseConv2D layers
def apply_selective_quantization(layer):
  # Skip custom layers and GroupNormalization - keep them as-is
  if isinstance(layer, (WaveformToLogMel, keras.layers.GroupNormalization)):
    return layer
  # Only quantize standard layers
  quantize_types = (keras.layers.Dense, keras.layers.Conv2D, keras.layers.DepthwiseConv2D)
  if isinstance(layer, quantize_types):
    return tfmot.quantization.keras.quantize_annotate_layer(layer)
  # Return all other layers unchanged
  return layer

print("\n=== APPLYING QUANTIZATION (keeping GroupNorm) ===")
# Clone and annotate
annotated_model = keras.models.clone_model(model, clone_function=apply_selective_quantization)

# Copy weights to annotated model
print("Copying weights to annotated model...")
for orig_layer, annot_layer in zip(model.layers, annotated_model.layers):
  orig_weights = orig_layer.get_weights()
  if not orig_weights:
    continue
  try:
    # For annotated layers, set weights on the inner layer
    if hasattr(annot_layer, 'layer'):
      annot_layer.layer.set_weights(orig_weights)
    else:
      annot_layer.set_weights(orig_weights)
  except Exception as e:
    print(f"Warning: Could not copy weights for {orig_layer.name}: {e}")

# Test annotated model before applying full quantization
annotated_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                        loss=keras.losses.SparseCategoricalCrossentropy(),
                        metrics=[keras.metrics.SparseCategoricalAccuracy()])
annot_eval = annotated_model.evaluate(test_ds_orig, return_dict=True)
print(f"After annotation (before quantize_apply) test accuracy: {annot_eval['sparse_categorical_accuracy']*100:.2f}%")

# Apply full quantization
with tfmot.quantization.keras.quantize_scope({'WaveformToLogMel': WaveformToLogMel}):
  quant_model = tfmot.quantization.keras.quantize_apply(annotated_model)

print("✓ Quantization applied successfully")
quant_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # type: ignore
                    loss=keras.losses.SparseCategoricalCrossentropy(),
                    metrics=[keras.metrics.SparseCategoricalAccuracy()])
quant_model.summary()

# Test initial quantized model accuracy
test_eval = quant_model.evaluate(test_ds_orig, return_dict=True)
print(f"Initial quantized model test accuracy: {test_eval['sparse_categorical_accuracy']*100:.2f}%")

# Carry out quantization-aware training for better quantized model accuracy
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = quant_model.fit(train_ds_orig, epochs=10000, validation_data=test_ds_orig, callbacks=[callback], verbose=2)  # type: ignore
quant_model.evaluate(test_ds_orig, return_dict=True)
tools.plot_training_history(history)
tools.plot_confusion_matrix(quant_model, test_ds_orig, dataset.idx_to_label)
quant_model.save(f'yamnetmini-quant-aware-{seed}.keras')

# Convert the quantized model to quantized TFLite format
tools.convert_keras_to_tflite(quant_model, f'yamnetmini-quantized-{seed}.tflite', True)

print(f"Quantized TFLite model accuracy: {tools.test_tflite_model(f'yamnetmini-quantized-{seed}.tflite', test_ds_orig)}")