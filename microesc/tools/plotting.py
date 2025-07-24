from typing import Dict
from .. import keras, PyDataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_waveform(waveform: np.ndarray, sample_rate: int, label: str | None = None):
  plt.figure(figsize=(10, 4))
  plt.plot(np.arange(len(waveform)) / sample_rate, waveform)
  plt.title('Audio Waveform' if not label else f'Audio Waveform: {label}')
  plt.xlabel('Time [s]')
  plt.ylabel('Amplitude')
  plt.ylim([-1.1, 1.1])
  plt.grid()
  plt.show()

def plot_spectrogram(waveform: np.ndarray, sample_rate: int, label: str | None = None):
  plt.figure(figsize=(10, 4))
  plt.specgram(waveform, Fs=sample_rate, NFFT=1024, noverlap=128, cmap='viridis', mode='magnitude', scale='linear')
  plt.title('Spectrogram' if not label else f'Spectrogram: {label}')
  plt.xlabel('Time [s]')
  plt.ylabel('Frequency [Hz]')
  plt.colorbar(label='Intensity [dB]')
  plt.ylim(0, sample_rate // 2)
  plt.show()

def plot_wave_and_spectrogram(waveform: np.ndarray, sample_rate: int, label: str | None = None):
  plt.figure(figsize=(12, 6))
  
  plt.subplot(2, 1, 1)
  plt.plot(np.arange(len(waveform)) / sample_rate, waveform)
  plt.title('Audio Waveform' if not label else f'Audio Waveform: {label}')
  plt.xlabel('Time [s]')
  plt.ylabel('Amplitude')
  plt.ylim([-1.1, 1.1])
  plt.grid()
  
  plt.subplot(2, 1, 2)
  plt.specgram(waveform, Fs=sample_rate, NFFT=1024, noverlap=128, cmap='viridis', mode='magnitude', scale='linear')
  plt.title('Spectrogram' if not label else f'Spectrogram: {label}')
  plt.xlabel('Time [s]')
  plt.ylabel('Frequency [Hz]')
  plt.ylim(0, sample_rate // 2)
  
  plt.tight_layout()
  plt.show()

def plot_training_history(history: keras.callbacks.History):
  metrics = history.history
  plt.figure(figsize=(10, 4))
  
  plt.subplot(1, 2, 1)
  plt.plot(history.epoch, metrics['loss'], label='Training')
  if 'val_loss' in metrics:
    plt.plot(history.epoch, metrics['val_loss'], label='Validation')
  plt.title('Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.ylim([0, max(max(metrics['loss']), max(metrics.get('val_loss', [0])))])
  plt.legend()
  
  plt.subplot(1, 2, 2)
  plt.plot(history.epoch, metrics['sparse_categorical_accuracy'], label='Training')
  if 'val_sparse_categorical_accuracy' in metrics:
    plt.plot(history.epoch, metrics['val_sparse_categorical_accuracy'], label='Validation')
  plt.title('Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0, 1])
  plt.legend()
  
  plt.tight_layout()
  plt.show()

def plot_confusion_matrix(trained_model: keras.Model, test_ds: PyDataset, idx_to_labels: Dict[int, str]):
  class_names = [idx_to_labels[i] for i in range(len(idx_to_labels))]
  y_true = np.concatenate([test_ds[i][1] for i in range(len(test_ds))], axis=0)
  y_pred = np.argmax(trained_model.predict(test_ds), axis=1)
  indices = np.stack([y_true, y_pred], axis=1)
  values = np.ones_like(y_pred, 'int32')
  confusion_matrix = np.zeros(np.stack([len(class_names), len(class_names)]), dtype=int)
  np.add.at(confusion_matrix, tuple(indices.reshape(-1, indices.shape[-1]).T), values.ravel())
  plt.figure(figsize=(12, 10))
  sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='rocket', xticklabels=class_names, yticklabels=class_names)
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.show()
