from .. import PyDataset
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
import math, random, librosa
import pandas as pd
import numpy as np

class AudioClip:

  class AudioData:

    def __init__(self, path: str, start_seconds: float, end_seconds: float | None, sample_rate: int):
      self.path = path
      self.sample_rate = sample_rate
      self.start_time = start_seconds
      self.end_time = end_seconds

    def __enter__(self):
      self.data, _ = librosa.load(self.path, sr=self.sample_rate, mono=True, offset=self.start_time, duration=(self.end_time - self.start_time) if self.end_time else None, dtype=np.float32)
      return self

    def __exit__(self, exc_type, exc_value, traceback):
      if exc_type is not None:
        print(exc_type, exc_value, traceback)
      if hasattr(self, 'data'):
        del self.data

  def __init__(self, label_idx: int, label: str, path: str, start_seconds: float, end_seconds: float | None, target_sample_rate_hz: int):
      self.label = label
      self.label_idx = label_idx
      self.audio = AudioClip.AudioData(path, start_seconds, end_seconds, target_sample_rate_hz)

  def __repr__(self):
    return f"AudioClip(label='{self.label}', path='{self.audio.path}', start_time={self.audio.start_time}, end_time={self.audio.end_time})"

  def __str__(self):
    return self.__repr__()


class KerasDataSet(PyDataset):

  def __init__(self, dataset: List[AudioClip], batch_size: int | None, **kwargs):
    super().__init__(**kwargs)
    self.batch_size = batch_size if batch_size is not None else 32
    self.dataset = dataset

  def __len__(self) -> int:
    return math.ceil(len(self.dataset) / self.batch_size)

  def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    batch_data, batch_labels = [], []
    low = idx * self.batch_size
    high = min(low + self.batch_size, len(self.dataset))
    for clip in self.dataset[low:high]:
      with clip.audio as audio:
        batch_data.append(audio.data)
        batch_labels.append(clip.label_idx)
    return np.array(batch_data), np.array(batch_labels)

  def on_epoch_end(self):
    random.shuffle(self.dataset)


class RavenProDataSet:

  def __init__(self,
               audio_path: str,
               annotation_path: str,
               target_sample_rate_hz: int,
               target_clip_length: float | None,
               training_split_percent: float = 0.8,
               uniform_classes_per_batch: bool = False,
               auto_create_negative_samples: bool = True):

    # Create structures to hold audio clips and labels
    self.clips, self.labels, self.label_counts = [], set(), defaultdict(int)
    self.label_to_idx, self.idx_to_label = {}, {}

    # Implicitly create an 'unlabeled' class if auto-creating negative samples
    negative_examples = []
    if auto_create_negative_samples:
      self.labels.add('unlabeled')
      self.label_to_idx['unlabeled'] = 0
      self.idx_to_label[0] = 'unlabeled'
      self.label_counts['unlabeled'] = 0

    # Iterate through all files in the annotation path
    for annotation_file in Path(annotation_path).glob("*.selections.txt"):

      # Locate the corresponding audio file based on the timestamp in the annotation file name
      timestamp = annotation_file.name[0:10] + "_" + annotation_file.name[11:13]  # makes "2025-05-02_12"
      for audio_file in Path(audio_path).glob("*.wav"):
        if timestamp in audio_file.name:

          # Parse the annotation file and extract labeled audio clips
          labeled_segments = []
          df = pd.read_csv(annotation_file, sep='\t')
          for start_time, end_time, label in df[['Begin Time (s)', 'End Time (s)', 'annotation']].to_numpy():

            # Add a new label mapping if the label doesn't exist
            if label not in self.labels:
              idx = len(self.label_to_idx)
              self.label_to_idx[label] = idx
              self.idx_to_label[idx] = label
              self.labels.add(label)

            # Center the audio clip within the target clip length, if specified
            if target_clip_length:
              if end_time - start_time < target_clip_length:
                start_time = max(0, start_time - ((target_clip_length - (end_time - start_time)) / 2.0))
              end_time = start_time + target_clip_length

            # Create an AudioClip instance and add it to the dataset
            clip = AudioClip(self.label_to_idx[label], label, str(audio_file.resolve()), start_time, end_time, target_sample_rate_hz)
            labeled_segments.append((start_time, end_time))
            self.label_counts[label] += 1
            self.clips.append(clip)

          # If auto-creating negative samples, add random unlabeled clips from the same audio file
          if auto_create_negative_samples:
            clip_length_seconds = librosa.get_duration(path=audio_file)
            if not target_clip_length:
              target_clip_length = 3.0 if clip_length_seconds > 3.0 else clip_length_seconds
            for start_time in np.arange(0.0, clip_length_seconds - target_clip_length + 0.001, target_clip_length):

              # Ensure the start and end times do not overlap with any labeled segments
              if not any(start_time < end and start_time + target_clip_length > begin for begin, end in labeled_segments):
                end_time = start_time + target_clip_length
                clip = AudioClip(0, 'unlabeled', str(audio_file.resolve()), start_time, end_time, target_sample_rate_hz)
                negative_examples.append(clip)

          # Break after processing the first matching audio file
          break

    # Augment dataset if uniform classes per batch are requested
    if uniform_classes_per_batch:
      max_count = max(self.label_counts.values())
      for label, count in self.label_counts.items():
        if count < max_count and label != 'unlabeled':
          for _ in range(max_count - count):
            clip = random.choice([c for c in self.clips if c.label == label])
            new_start_time = max(0, random.uniform(clip.audio.start_time - 0.150, clip.audio.start_time + 0.150))
            new_end_time = new_start_time + (clip.audio.end_time - clip.audio.start_time) if clip.audio.end_time else None
            self.clips.append(AudioClip(clip.label_idx, clip.label, clip.audio.path, new_start_time, new_end_time, clip.audio.sample_rate))
            self.label_counts[label] += 1

    # Randomly add negative examples if auto-creating negative samples
    if auto_create_negative_samples:
      max_count = max(self.label_counts.values())
      random.shuffle(negative_examples)
      for i in range(max_count):
        clip = negative_examples[i % len(negative_examples)]
        self.clips.append(clip)
        self.label_counts['unlabeled'] += 1

    # Shuffle the clips and split them into a training and test set
    random.shuffle(self.clips)
    self.training_clips = self.clips[:int(len(self.clips) * training_split_percent)]
    self.test_clips = self.clips[int(len(self.clips) * training_split_percent):]

  def __len__(self) -> int:
    return len(self.clips)

  def __getitem__(self, idx: int) -> AudioClip:
    return self.clips[idx]

  def summary(self):
    print("\n\033[1mDataset summary:\033[0m")
    print(f"   \033[1mTotal clips:\033[0m {len(self.clips)}")
    print(f"   \033[1mTraining clips:\033[0m {len(self.training_clips)}")
    print(f"   \033[1mTest clips:\033[0m {len(self.test_clips)}")
    print("   \033[1mLabels and Counts:\033[0m")
    for label, count in sorted(self.label_counts.items()):
      print(f"      {label}: {count}")
    print()

  def train_dataset(self, batch_size: int | None = None, **kwargs) -> KerasDataSet:
    return KerasDataSet(self.training_clips, batch_size, **kwargs)

  def test_dataset(self, batch_size: int | None = None, **kwargs) -> KerasDataSet:
    return KerasDataSet(self.test_clips, batch_size, **kwargs)
