from microesc.detection.SpectralFluxDetector import SpectralFluxDetector
import numpy as np

np.set_printoptions(suppress=True, precision=3)
detector = SpectralFluxDetector(9.0, 8000, 512, 256, False, 150.0, 1800.0, 0.1)

print('Detected events in tests/audio/9_mm_gunshot-mike-koenig-123.wav:')
for event in detector.detect_events('tests/audio/9_mm_gunshot-mike-koenig-123.wav'):
  print(event)
print('Detected events in tests/audio/gunshots-345584.wav:')
for event in detector.detect_events('tests/audio/gunshots-345584.wav'):
  print(event)
print('Detected events in tests/audio/birds-1-34495-A.wav:')
for event in detector.detect_events('tests/audio/birds-1-34495-A.wav'):
  print(event)
print('Detected events in tests/audio/gun-2018-03-11.030235.4480-0.471244.wav:')
for event in detector.detect_events('tests/audio/gun-2018-03-11.030235.4480-0.471244.wav'):
  print(event)
print('Detected events in tests/audio/gun-2018-03-17.040928.7220-0.125898.wav:')
for event in detector.detect_events('tests/audio/gun-2018-03-17.040928.7220-0.125898.wav'):
  print(event)
print('Detected events in tests/audio/gun-2018-03-11.030236.5840-0.208148.wav:')
for event in detector.detect_events('tests/audio/gun-2018-03-11.030236.5840-0.208148.wav'):
  print(event)
