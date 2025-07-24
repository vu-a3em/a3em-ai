__version__ = "0.1.0"

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # TODO: remove this when keras is updated
import tf_keras as keras   # TODO: import keras
from tf_keras.utils import Sequence as PyDataset  # TODO: from keras.utils import PyDataset
from . import ops_compat as ops  # TODO: DELETE IN FAVOR OF from keras import ops (also delete ops_compat.py)
