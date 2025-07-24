from abc import ABCMeta, abstractmethod
from typing import List

class EventDetector(metaclass=ABCMeta):
    
  @abstractmethod
  def __init__(self, name: str):
    self.name = name

  @abstractmethod
  def detect_events(self, audio_path: str) -> List[float]:
    pass
