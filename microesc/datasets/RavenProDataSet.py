from .. import PyDataset
from typing import List, Tuple
import glob, os, math, random, librosa
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
               base_path: str,
               target_sample_rate_hz: int,
               training_split_percent: float = 0.8,
               uniform_classes_per_batch: bool = False):

    # Create structures to hold audio clips and labels
    self.clips, self.labels = [], set()
    self.label_to_idx, self.idx_to_label = {}, {}

    # Iterate through all files in the base path and do something with them
    # TODO

    # Shuffle the clips and split them into a training and test set
    random.shuffle(self.clips)
    self.training_clips = self.clips[:int(len(self.clips) * training_split_percent)]
    self.test_clips = self.clips[int(len(self.clips) * training_split_percent):]

  def __len__(self) -> int:
    return len(self.clips)

  def __getitem__(self, idx: int) -> AudioClip:
    return self.clips[idx]

  def _parse_annotations(self, annotation_file: str) -> List[float]:
    metadata = []
    with open(annotation_file, 'r') as file:
      for line in file:
        # TODO: Parse the line to extract the annotation metadata
        pass
    return metadata

  def summary(self):
    label_counts = dict.fromkeys(self.labels, 0)
    for clip in self.clips:
      label_counts[clip.label] += 1
    print("\n\033[1mDataset summary:\033[0m")
    print(f"   \033[1mTotal clips:\033[0m {len(self.clips)}")
    print(f"   \033[1mTraining clips:\033[0m {len(self.training_clips)}")
    print(f"   \033[1mTest clips:\033[0m {len(self.test_clips)}")
    print("   \033[1mLabels and Counts:\033[0m")
    for label, count in sorted(label_counts.items()):
      print(f"      {label}: {count}")
    print()

  def train_dataset(self, batch_size: int | None = None, **kwargs) -> KerasDataSet:
    return KerasDataSet(self.training_clips, batch_size, **kwargs)

  def test_dataset(self, batch_size: int | None = None, **kwargs) -> KerasDataSet:
    return KerasDataSet(self.test_clips, batch_size, **kwargs)
