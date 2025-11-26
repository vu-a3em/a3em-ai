import os
from microesc.classification.Yamnet import YamnetParams

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_ROOT = os.path.join(BASE_DIR, "user_uploads")

# Yamnet Constants (Fixed)
params = YamnetParams()
TARGET_SAMPLE_RATE = params.sample_rate
TARGET_CLIP_LENGTH = params.patch_window_seconds + params.stft_window_seconds - params.stft_hop_seconds
PATCH_HOP_SECONDS = params.patch_hop_seconds

# Default Training Hyperparameters
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 10
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_BACKGROUND_RATIO = 0.3
