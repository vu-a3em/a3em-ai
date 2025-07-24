from .. import PyDataset
from collections import defaultdict
from ..detection import EventDetector
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
      #with self.audio as audio:
      #  self._compute_mfcc(audio)
      #  self._compute_zcr(audio)

  def __repr__(self):
    return f"AudioClip(label='{self.label}', path='{self.audio.path}', start_time={self.audio.start_time}, end_time={self.audio.end_time})"

  def __str__(self):
    return self.__repr__()

  def _compute_mfcc(self, audio: AudioData):
    melspectrogram = librosa.feature.melspectrogram(y=audio.data, sr=audio.sample_rate)
    logamplitude = librosa.amplitude_to_db(melspectrogram, ref=np.max)
    self.mfcc = librosa.feature.mfcc(S=logamplitude, sr=audio.sample_rate, n_mfcc=13).transpose()

  def _compute_zcr(self, audio: AudioData):
    self.zcr = librosa.feature.zero_crossing_rate(y=audio.data).transpose().flatten()


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


class DirectoryDataSet:

  def __init__(self,
               base_path: str,
               target_sample_rate_hz: int,
               target_clip_length: float | None,
               training_split_percent: float = 0.8,
               uniform_classes_per_batch: bool = False,
               event_start_offset: float = 0.0,
               event_detector: EventDetector | None = None,
               event_detector_match_metadata_leeway_seconds: float | None = None):

    # Create structures to hold audio clips and labels
    self.clips, self.labels, self.label_counts = [], set(), defaultdict(int)
    self.label_to_idx, self.idx_to_label = {}, {}

    # Iterate through all directories in the base path (which should correspond to class labels)
    for idx, dir in enumerate(glob.glob(os.path.join(os.path.abspath(base_path), '*'))):

      # Get the label from the directory name and create a label index
      label = os.path.basename(dir)
      self.labels.add(label)
      self.label_to_idx[label] = idx
      self.idx_to_label[idx] = label

      # Iterate through all audio files in the directory
      for file in glob.glob(os.path.join(dir, '**'), recursive=True):
        if file.lower().endswith(('.wav', '.mp3', '.ogg', '.m4a', '.aac')):
          audio_length_seconds = librosa.get_duration(path=file)

          # Search for a metadata file if requested
          metadata = None
          if (event_detector and event_detector_match_metadata_leeway_seconds is not None) or target_clip_length:
            metadata_file = file[:-4] + '.meta'
            if os.path.exists(metadata_file):
              metadata = self._parse_metadata(metadata_file)
            else:
              print(f"Metadata file for {file} does not exist, ignoring metadata...")

          # Create audio clips for each file based on either event detection output or fixed-length segmentation
          if event_detector:
            for event_onset in event_detector.detect_events(file):

              # Check if the event onset matches any metadata-specified onsets
              if not metadata or np.isclose(metadata, event_onset, atol=event_detector_match_metadata_leeway_seconds).any():  # type: ignore

                # Place the start of the event at the specified offset within the clip
                event_onset -= event_start_offset
                event_end_time = (event_onset + target_clip_length) if target_clip_length else None

                # Only create a clip if the start and end times are valid
                if event_onset >= 0.0 and (not event_end_time or event_end_time <= audio_length_seconds):
                  clip = AudioClip(idx, label, file, event_onset, event_end_time, target_sample_rate_hz)
                  self.label_counts[label] += 1
                  self.clips.append(clip)

          elif target_clip_length:

            # If metadata is provided, create clips based on the metadata
            if metadata:
              for start_time in metadata:
                event_onset = start_time - event_start_offset
                event_end_time = event_onset + target_clip_length
                if event_onset >= 0.0 and event_end_time <= audio_length_seconds:
                  clip = AudioClip(idx, label, file, event_onset, event_end_time, target_sample_rate_hz)
                  self.label_counts[label] += 1
                  self.clips.append(clip)

            # Otherwise, create clips based on fixed-length segmentation
            else:
              for start_time in np.arange(0, audio_length_seconds, target_clip_length):
                event_end_time = start_time + target_clip_length
                if event_end_time <= audio_length_seconds:
                  clip = AudioClip(idx, label, file, start_time, event_end_time, target_sample_rate_hz)
                  self.label_counts[label] += 1
                  self.clips.append(clip)

          else:
            clip = AudioClip(idx, label, file, 0.0, target_clip_length, target_sample_rate_hz)
            self.label_counts[label] += 1
            self.clips.append(clip)

    # Augment dataset if uniform classes per batch is requested
    if uniform_classes_per_batch:
      max_count = max(self.label_counts.values())
      for label, count in self.label_counts.items():
        if count < max_count:
          for _ in range(max_count - count):
            clip = random.choice([c for c in self.clips if c.label == label])
            new_start_time = max(0, random.uniform(clip.audio.start_time - 0.150, clip.audio.start_time + 0.150))
            new_end_time = new_start_time + (clip.audio.end_time - clip.audio.start_time) if clip.audio.end_time else None
            self.clips.append(AudioClip(clip.label_idx, clip.label, clip.audio.path, new_start_time, new_end_time, clip.audio.sample_rate))
            self.label_counts[label] += 1

    # Shuffle the clips and split them into a training and test set
    random.shuffle(self.clips)
    self.training_clips = self.clips[:int(len(self.clips) * training_split_percent)]
    self.test_clips = self.clips[int(len(self.clips) * training_split_percent):]

  def __len__(self) -> int:
    return len(self.clips)

  def __getitem__(self, idx: int) -> AudioClip:
    return self.clips[idx]

  def _parse_metadata(self, metadata_file: str) -> List[float]:
    metadata = []
    with open(metadata_file, 'r') as file:
      for line in file:
        tokens = line.split(',')
        if len(tokens) >= 2 and tokens[-1].strip().lower() != 'ignore' and tokens[-1].strip().lower() != 'unknown':
          metadata.append(float(tokens[-2].strip()))
    return metadata

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
