from .. import PyDataset
from collections import defaultdict
from ..detection import EventDetector
from typing import List, Tuple
import glob, os, math, random, librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from .DirectoryDataSet import AudioClip

class KerasDataSet(PyDataset):
  """
  A tf.data.Dataset that loads audio clips from files on demand.
  """

  def __init__(self, dataset: List[AudioClip], batch_size: int | None, background_clips: List[AudioClip] = None, background_to_event_ratio: float = 0.0, **kwargs):
    super().__init__(**kwargs)
    self.batch_size = batch_size if batch_size is not None else 32
    self.dataset = dataset
    self.background_clips = background_clips if background_clips is not None else []
    self.background_to_event_ratio = background_to_event_ratio

  def __len__(self) -> int:
    return math.ceil(len(self.dataset) / self.batch_size)

  def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    batch_data, batch_labels = [], []
    low = idx * self.batch_size
    high = min(low + self.batch_size, len(self.dataset))
    for clip in self.dataset[low:high]:
      with clip.audio as audio:
        event_data = audio.data
        # scale event audio
        scale = np.max(np.abs(event_data)) if np.max(np.abs(event_data)) > 0 else 1.0
        event_data = event_data / scale

        # Mix with background if available and ratio > 0
        if self.background_clips and (isinstance(self.background_to_event_ratio, list) or self.background_to_event_ratio > 0.0):
          bg_clip = random.choice(self.background_clips)
          with bg_clip.audio as bg_audio:
            bg_data = bg_audio.data
            # Pad/trim background to match event length
            if len(bg_data) < len(event_data):
              bg_data = np.pad(bg_data, (0, len(event_data) - len(bg_data)), mode='constant')
            else:
              bg_data = bg_data[:len(event_data)]
            # scale background audio
            bg_scale = np.max(np.abs(bg_data)) if np.max(np.abs(bg_data)) > 0 else 1.0
            bg_data = bg_data / bg_scale

            # Determine mixing ratio
            if isinstance(self.background_to_event_ratio, list):
              assert len(self.background_to_event_ratio) == 2 and self.background_to_event_ratio[0] <= self.background_to_event_ratio[1], "background_to_event_ratio list must have two elements [min, max] with min <= max"
              ratio = random.uniform(self.background_to_event_ratio[0], self.background_to_event_ratio[1])
            else:
              ratio = self.background_to_event_ratio

            # Mix event and background audio
            event_data = event_data * (1 - ratio) + bg_data * ratio

        batch_data.append(event_data)
        batch_labels.append(clip.label_idx)
    return np.array(batch_data), np.array(batch_labels)

  def on_epoch_end(self):
    random.shuffle(self.dataset)

def _process_audio_file(args):
  """
  Processes a single audio file to extract AudioClip instances based on event detection or fixed-length segments.
  Used with ThreadPoolExecutor for parallel processing.
  """
  file, idx, label, target_sample_rate_hz, target_clip_length, event_start_offset, event_detector, event_detector_match_metadata_leeway_seconds, parse_metadata_func = args
  clips = []
  try:
    audio_length_seconds = librosa.get_duration(path=file)
    metadata = None
    if (event_detector and event_detector_match_metadata_leeway_seconds is not None) or target_clip_length:
      metadata_file = file[:-4] + '.meta'
      if os.path.exists(metadata_file):
        metadata = parse_metadata_func(metadata_file)

    if event_detector:
      for event_onset in event_detector.detect_events(file):
        if not metadata or np.isclose(metadata, event_onset, atol=event_detector_match_metadata_leeway_seconds).any():
          event_onset -= event_start_offset
          event_end_time = (event_onset + target_clip_length) if target_clip_length else None
          if event_onset >= 0.0 and (not event_end_time or event_end_time <= audio_length_seconds):
            clips.append(AudioClip(idx, label, file, event_onset, event_end_time, target_sample_rate_hz))
    elif target_clip_length:
      if metadata:
        for start_time in metadata:
          event_onset = start_time - event_start_offset
          event_end_time = event_onset + target_clip_length
          if event_onset >= 0.0 and event_end_time <= audio_length_seconds:
            clips.append(AudioClip(idx, label, file, event_onset, event_end_time, target_sample_rate_hz))
      else:
        for start_time in np.arange(0, audio_length_seconds, target_clip_length):
          event_end_time = start_time + target_clip_length
          if event_end_time <= audio_length_seconds:
            clips.append(AudioClip(idx, label, file, start_time, event_end_time, target_sample_rate_hz))
    else:
      clips.append(AudioClip(idx, label, file, 0.0, target_clip_length, target_sample_rate_hz))
  except Exception as e:
    print(f"Error processing {file}: {e}")
  return clips

class AugmentedDirectoryDataSet:
  """
  Loads audio clips from a directory structure where each subdirectory represents a class label (sub-subdirectories are flattened).
  """

  def __init__(self,
               base_path: str,
               target_sample_rate_hz: int,
               target_clip_length: float | None,
               training_split_percent: float = 0.8,
               uniform_classes_per_batch: bool = False,
               event_start_offset: float = 0.0,
               event_detector: EventDetector | None = None,
               event_detector_match_metadata_leeway_seconds: float | None = None,
               ignore_directories: List[str] = [],
               background_classes: List[str] = [],
               background_to_event_ratio: float | List[float] = 0.0,
               max_samples_per_class: int | None = None,
               ):


        # Create structures to hold audio clips and labels
        self.clips, self.labels, self.label_counts = [], set(), defaultdict(int)
        self.label_to_idx, self.idx_to_label = {}, {}
        self.background_clips = []
        
        self.background_to_event_ratio = background_to_event_ratio

        files_to_process = []
        idx = 0
        for dir in glob.glob(os.path.join(os.path.abspath(base_path), '*')):
            if not os.path.isdir(dir) or os.path.basename(dir) in ignore_directories:
                print(f"Ignoring directory: {dir}")
                continue
            label = os.path.basename(dir)
            if label in background_classes:
                # Collect background clips separately
                for file in glob.glob(os.path.join(dir, '**'), recursive=True):
                    if file.lower().endswith(('.wav', '.mp3', '.ogg', '.m4a', '.aac')):
                        files_to_process.append((
                            file, None, label, target_sample_rate_hz, target_clip_length,
                            event_start_offset, event_detector, event_detector_match_metadata_leeway_seconds,
                            self._parse_metadata
                        ))
                continue  # Do not add background labels to main label structures

            self.labels.add(label)
            self.label_to_idx[label] = idx
            self.idx_to_label[idx] = label

            dir_files = []
            for file in glob.glob(os.path.join(dir, '**'), recursive=True):
                if file.lower().endswith(('.wav', '.mp3', '.ogg', '.m4a', '.aac')):
                    dir_files.append((
                        file, idx, label, target_sample_rate_hz, target_clip_length,
                        event_start_offset, event_detector, event_detector_match_metadata_leeway_seconds,
                        self._parse_metadata
                    ))
            
            files_to_process.extend(dir_files)
            idx += 1

        # Parallel processing using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_process_audio_file, args) for args in files_to_process]
            for future in as_completed(futures):
                for clip in future.result():
                    if clip.label in background_classes:
                        self.background_clips.append(clip)
                    else:
                        self.clips.append(clip)
                        self.label_counts[clip.label] += 1

        # Limit number of samples per class if specified
        if max_samples_per_class:
            new_clips = []
            for label in self.labels:
                label_clips = [c for c in self.clips if c.label == label]
                if len(label_clips) > max_samples_per_class:
                    label_clips = random.sample(label_clips, max_samples_per_class)
                new_clips.extend(label_clips)
                self.label_counts[label] = len(label_clips)
            self.clips = new_clips

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
    """
    Prints a summary of the dataset including total clips, training/test split, and label counts.
    """
    print("\n\033[1mDataset summary:\033[0m")
    print(f"   \033[1mTotal clips:\033[0m {len(self.clips)}")
    print(f"   \033[1mTraining clips:\033[0m {len(self.training_clips)}")
    print(f"   \033[1mTest clips:\033[0m {len(self.test_clips)}")
    print(f"   \033[1mLabels (total {len(self.label_counts)}) and Counts:\033[0m")
    for label, count in sorted(self.label_counts.items()):
      print(f"      {label}: {count}")
    print()

  def train_dataset(self, batch_size: int | None = None, **kwargs) -> KerasDataSet:
    return KerasDataSet(self.training_clips, batch_size, background_clips=self.background_clips, background_to_event_ratio=self.background_to_event_ratio, **kwargs)

  def test_dataset(self, batch_size: int | None = None, **kwargs) -> KerasDataSet:
    return KerasDataSet(self.test_clips, batch_size, background_clips=self.background_clips, background_to_event_ratio=self.background_to_event_ratio, **kwargs)
