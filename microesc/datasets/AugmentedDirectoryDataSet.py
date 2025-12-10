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

  def __init__(self, dataset: List[AudioClip], batch_size: int | None, background_clips: List[AudioClip] = None, background_to_event_ratio: float = 0.0, expected_samples: int = None,  **kwargs):
    super().__init__(**kwargs)
    self.batch_size = batch_size if batch_size is not None else 32
    self.dataset = dataset
    self.background_clips = background_clips if background_clips is not None else []
    self.background_to_event_ratio = background_to_event_ratio
    self.expected_samples = expected_samples

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

        if self.expected_samples and len(event_data) < self.expected_samples:
          # print(f"Padding audio from {len(event_data)} to expected {self.expected_samples} samples.")
          event_data = np.pad(event_data, (0, self.expected_samples - len(event_data)), mode='constant')

        batch_data.append(event_data)
        batch_labels.append(clip.label_idx)
    try:
        return np.array(batch_data), np.array(batch_labels)
    except Exception as e:
        print(f"Error converting batch data to numpy arrays: {e}")
        print(f"Batch data lengths: {[len(d) for d in batch_data]}")
        print(f"Batch label lengths: {[l for l in batch_labels]}")
        raise e

  def on_epoch_end(self):
    random.shuffle(self.dataset)

  def summary(self):
    """
    Prints a summary of the dataset including total clips and batch size.
    """
    print("\n\033[1mKerasDataSet summary:\033[0m")
    print(f"   \033[1mTotal clips:\033[0m {len(self.dataset)}")
    print(f"   \033[1mBatch size:\033[0m {self.batch_size}")
    if self.background_clips:
      print(f"   \033[1mBackground clips:\033[0m {len(self.background_clips)}")
      print(f"   \033[1mBackground to event ratio:\033[0m {self.background_to_event_ratio}")

def _process_audio_file(args):
  """
  Processes a single audio file to extract AudioClip instances based on event detection or fixed-length segments.
  Used with ThreadPoolExecutor for parallel processing.
  """
  file, idx, label, target_sample_rate_hz, target_clip_length, event_start_offset, event_detector, event_detector_match_metadata_leeway_seconds, use_metadata, parse_metadata_func = args
  clips = []
  try:
    audio_length_seconds = librosa.get_duration(path=file)
    metadata = None
    if (event_detector and event_detector_match_metadata_leeway_seconds is not None) or target_clip_length:
      metadata_file = file[:-4] + '.meta'
      if os.path.exists(metadata_file):
        metadata = parse_metadata_func(metadata_file)
        metadata_starts = np.array([start for start, _ in metadata])

    # Use the clipping from the metadata if requested
    if use_metadata and metadata:
      for start_time, metadata_label in metadata:
        event_onset = start_time - event_start_offset
        event_end_time = event_onset + target_clip_length if target_clip_length else None

        if event_onset >= 0.0 and (not event_end_time or event_end_time <= audio_length_seconds):
          clips.append(AudioClip(idx, metadata_label, file, event_onset, event_end_time, target_sample_rate_hz))
    else:
      if event_detector:
        for event_onset in event_detector.detect_events(file):
          if not metadata_starts or np.isclose(metadata_starts, event_onset, atol=event_detector_match_metadata_leeway_seconds).any():
            event_onset -= event_start_offset
            event_end_time = (event_onset + target_clip_length) if target_clip_length else None
            if event_onset >= 0.0 and (not event_end_time or event_end_time <= audio_length_seconds):

              metadata_label = None

              if metadata and metadata_starts is not None:
                # Check if the event onset matches any metadata start time
                if np.isclose(metadata_starts, event_onset, atol=event_detector_match_metadata_leeway_seconds).any():
                  metadata_label = next((label for start, label in metadata if np.isclose(start, event_onset, atol=event_detector_match_metadata_leeway_seconds)), None)

              if metadata_label is None:
                metadata_label = label  # Use the main label if no metadata match
              
              clips.append(AudioClip(idx, metadata_label, file, event_onset, event_end_time, target_sample_rate_hz))
      elif target_clip_length:
        if metadata:
          for start_time, metadata_label in metadata:
            event_onset = start_time - event_start_offset
            event_end_time = event_onset + target_clip_length
            if event_onset >= 0.0 and event_end_time <= audio_length_seconds:
              clips.append(AudioClip(idx, metadata_label, file, event_onset, event_end_time, target_sample_rate_hz))
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
  Supports adding additional labeled or background clips after initialization.
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
               use_metadata: bool = False,
               ):

        self.training_split_percent = training_split_percent
        self.uniform_classes_per_batch = uniform_classes_per_batch
        self.max_samples_per_class = max_samples_per_class

        # Create structures to hold audio clips and labels
        self.clips, self.labels, self.label_counts = [], set(), defaultdict(int)
        self.label_to_idx, self.idx_to_label = {}, {}
        self.background_clips = []

        self.background_to_event_ratio = background_to_event_ratio
        self.expected_samples = int(target_clip_length * target_sample_rate_hz) if target_clip_length else None

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
                            use_metadata, self._parse_metadata
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
                        use_metadata,
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
                    elif clip.label is not None and clip.label in self.labels:
                        # Only add clips with valid labels
                        self._add_clip_internal(clip)

        # Limit number of samples per class if specified
        if self.max_samples_per_class:
            self._apply_max_samples_per_class()

        # Augment dataset if uniform classes per batch is requested
        if self.uniform_classes_per_batch:
            self._augment_uniform_classes()

        # Initial split
        self._split_train_test()

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
          metadata.append((float(tokens[-2].strip()), tokens[-1].strip()))
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
    return KerasDataSet(self.training_clips, batch_size, background_clips=self.background_clips, background_to_event_ratio=self.background_to_event_ratio, expected_samples=self.expected_samples, **kwargs)

  def test_dataset(self, batch_size: int | None = None, **kwargs) -> KerasDataSet:
    return KerasDataSet(self.test_clips, batch_size, background_clips=self.background_clips, background_to_event_ratio=self.background_to_event_ratio, expected_samples=self.expected_samples, **kwargs)

  def _add_clip_internal(self, clip: AudioClip) -> None:
    """Internal helper to register a non-background clip and update counts and label indices."""
    # Ensure mapping exists for this label
    if clip.label not in self.label_to_idx:
      new_idx = len(self.label_to_idx)
      self.label_to_idx[clip.label] = new_idx
      self.idx_to_label[new_idx] = clip.label
      self.labels.add(clip.label)
    # Normalize label_idx
    clip.label_idx = self.label_to_idx[clip.label]
    self.clips.append(clip)
    self.label_counts[clip.label] += 1

  def _split_train_test(self) -> None:
    """Create or refresh training/test split from current clips."""
    if not self.clips:
      self.training_clips, self.test_clips = [], []
      return
    random.shuffle(self.clips)
    split_idx = int(len(self.clips) * self.training_split_percent)
    self.training_clips = self.clips[:split_idx]
    self.test_clips = self.clips[split_idx:]

  def _apply_max_samples_per_class(self) -> None:
    """Limit samples per class according to self.max_samples_per_class."""
    if not self.max_samples_per_class:
      return
    new_clips = []
    new_counts = defaultdict(int)
    for label in self.labels:
      label_clips = [c for c in self.clips if c.label == label]
      if len(label_clips) > self.max_samples_per_class:
        label_clips = random.sample(label_clips, self.max_samples_per_class)
      new_clips.extend(label_clips)
      new_counts[label] = len(label_clips)
    self.clips = new_clips
    self.label_counts = new_counts

  def _augment_uniform_classes(self) -> None:
    """Upsample classes so each has the same number of samples as the largest class."""
    if not self.label_counts:
      return
    max_count = max(self.label_counts.values())
    for label, count in list(self.label_counts.items()):
      if count < max_count:
        candidates = [c for c in self.clips if c.label == label]
        if not candidates:
          continue
        needed = max_count - count
        for _ in range(needed):
          base = random.choice(candidates)
          original = base.audio
          length = (original.end_time - original.start_time) if original.end_time else 0.0
          if length <= 0.0:
            aug = AudioClip(base.label_idx, base.label, original.path, original.start_time, original.end_time, original.sample_rate)
          else:
            shift = random.uniform(-0.150, 0.150)
            new_start = max(0.0, original.start_time + shift)
            new_end = new_start + length
            aug = AudioClip(base.label_idx, base.label, original.path, new_start, new_end, original.sample_rate)
          self.clips.append(aug)
          self.label_counts[label] += 1

  # Public API: incremental updates

  def add_clips_for_label(self, label: str, clips: List[AudioClip], *, is_background: bool = False, resplit: bool = True) -> None:
    """
    Add pre-constructed AudioClip instances for a given label.
    - If is_background is True, clips are added to background_clips only.
    - Otherwise, label is treated as a normal class label (created if needed).
    Generic enough for "Other" or any other label.
    """
    if is_background:
      self.background_clips.extend(clips)
      return

    # Ensure label index exists
    if label not in self.label_to_idx:
      new_idx = len(self.label_to_idx)
      self.label_to_idx[label] = new_idx
      self.idx_to_label[new_idx] = label
      self.labels.add(label)

    for clip in clips:
      clip.label = label
      clip.label_idx = self.label_to_idx[label]
      self._add_clip_internal(clip)

    if self.max_samples_per_class:
      self._apply_max_samples_per_class()

    if self.uniform_classes_per_batch:
      self._augment_uniform_classes()

    if resplit:
      self._split_train_test()

  def add_class_from_directory(self,
                               label: str,
                               path: str,
                               target_sample_rate_hz: int,
                               target_clip_length: float | None,
                               event_start_offset: float = 0.0,
                               event_detector: EventDetector | None = None,
                               event_detector_match_metadata_leeway_seconds: float | None = None,
                               use_metadata: bool = False,
                               max_samples: int | None = None,
                               resplit: bool = True) -> None:
    """
    Scan a directory (recursively) and add all matching audio clips as a new or existing label.
    Generic utility; can be used to add an "Other" class after initialization.

    max_samples (optional): if provided, randomly subsample at most this many
    source files before processing into AudioClip instances.
    """
    # Ensure label index exists
    if label not in self.label_to_idx:
      new_idx = len(self.label_to_idx)
      self.label_to_idx[label] = new_idx
      self.idx_to_label[new_idx] = label
      self.labels.add(label)

    # Collect all candidate audio files
    all_files = [
      file for file in glob.glob(os.path.join(os.path.abspath(path), '**'), recursive=True)
      if file.lower().endswith(('.wav', '.mp3', '.ogg', '.m4a', '.aac'))
    ]
    if not all_files:
      return

    # Optionally subsample sources
    if max_samples is not None and len(all_files) > max_samples:
      all_files = random.sample(all_files, max_samples)

    # Build args for processing
    files_to_process = [
      (
        file,
        self.label_to_idx[label],
        label,
        target_sample_rate_hz,
        target_clip_length,
        event_start_offset,
        event_detector,
        event_detector_match_metadata_leeway_seconds,
        use_metadata,
        self._parse_metadata,
      )
      for file in all_files
    ]

    # Parallel processing and registration
    with ThreadPoolExecutor() as executor:
      futures = [executor.submit(_process_audio_file, args) for args in files_to_process]
      for future in as_completed(futures):
        for clip in future.result():
          if clip.label is not None and clip.label in self.labels:
            self._add_clip_internal(clip)

    # Allow override of max_samples_per_class for this addition
    if self.max_samples_per_class and not max_samples:
      self._apply_max_samples_per_class()

    if self.uniform_classes_per_batch:
      self._augment_uniform_classes()

    if resplit:
      self._split_train_test()

  def add_background_from_directory(self,
                                    path: str,
                                    target_sample_rate_hz: int,
                                    target_clip_length: float | None,
                                    event_start_offset: float = 0.0,
                                    event_detector: EventDetector | None = None,
                                    event_detector_match_metadata_leeway_seconds: float | None = None,
                                    use_metadata: bool = False) -> None:
    """
    Add additional background-only clips from a directory.
    These clips are only used for mixing (not as a predicted class).
    """
    files_to_process = []
    dummy_label = "__background__"
    for file in glob.glob(os.path.join(os.path.abspath(path), '**'), recursive=True):
      if file.lower().endswith(('.wav', '.mp3', '.ogg', '.m4a', '.aac')):
        files_to_process.append((
          file,
          None,
          dummy_label,
          target_sample_rate_hz,
          target_clip_length,
          event_start_offset,
          event_detector,
          event_detector_match_metadata_leeway_seconds,
          use_metadata,
          self._parse_metadata,
        ))

    if not files_to_process:
      return

    with ThreadPoolExecutor() as executor:
      futures = [executor.submit(_process_audio_file, args) for args in files_to_process]
      for future in as_completed(futures):
        for clip in future.result():
          self.background_clips.append(clip)
