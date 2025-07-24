from ..detection import EventDetector
from typing import List
import numpy as np
import librosa

class SpectralFluxDetector(EventDetector):

  def __init__(self, threshold: float, sample_rate: int, fft_length: int, hop_length: int, use_power_spectrum: bool, min_frequency: float | None, max_frequency: float | None, min_seconds_between_events: float):
    super().__init__('SpectralFluxDetector')
    self.min_threshold = threshold
    self.sample_rate = sample_rate
    self.fft_length = fft_length
    self.hop_length = hop_length
    self.use_power_spectrum = use_power_spectrum
    self.min_frequency = min_frequency
    self.max_frequency = max_frequency
    self.min_seconds_between_events = min_seconds_between_events

  def detect_events(self, audio_path: str) -> List[float]:
    # Load audio data with the proper sampling rate and a single audio channel and compute the spectrogram
    data, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True, dtype=np.float32)
    spectrum = np.abs(librosa.stft(data, n_fft=self.fft_length, hop_length=self.hop_length, center=False))

    # Convert to power spectrum if required
    if self.use_power_spectrum:
      spectrum = spectrum ** 2

    # Ignore frequencies outside the specified range
    min_bin = int(np.floor(self.min_frequency * self.fft_length / self.sample_rate)) if self.min_frequency else 0
    max_bin = int(np.ceil(self.max_frequency * self.fft_length / self.sample_rate)) if self.max_frequency else spectrum.shape[0]
    spectrum = spectrum[min_bin:max_bin, :]

    # Compute the spectral flux for all time frames
    flux = np.insert(np.sum(np.clip(np.diff(spectrum, axis=1), a_min=0, a_max=None), axis=0), 0, 0)
    
    # Search for events based on spectral flux peaks
    events = []
    threshold = self.min_threshold
    for i in range(1, len(flux)):
      if flux[i] >= threshold and flux[i] >= 2 * flux[i-1] and (not events or (self.hop_length / self.sample_rate * (i + 1) - events[-1]) >= self.min_seconds_between_events):
         events.append(self.hop_length / self.sample_rate * (i + 1))
      threshold = max(max(flux[i], threshold * 0.9), self.min_threshold)
    return events
