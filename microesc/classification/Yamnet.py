from dataclasses import dataclass, asdict
from .. import keras, ops
import math

@dataclass()
class YamnetParams:
  sample_rate: int = 16000
  stft_window_seconds: float = 0.025
  stft_hop_seconds: float = 0.010
  mel_bands: int = 64
  mel_min_hz: float = 125.0
  mel_max_hz: float = 7500.0
  log_offset: float = 0.001
  patch_window_seconds: float = 0.96
  patch_hop_seconds: float = 0.48
  num_classes: int = 521
  conv_padding: str = 'same'
  batchnorm_center: bool = True
  batchnorm_scale: bool = False
  batchnorm_epsilon: float = 1e-4
  classifier_activation: str = 'softmax'

  @property
  def patch_frames(self) -> int:
    return int(round(self.patch_window_seconds / self.stft_hop_seconds))

  @property
  def patch_bands(self) -> int:
    return self.mel_bands


@keras.saving.register_keras_serializable(package='yamnet')
class WaveformToLogMel(keras.layers.Layer):  # TODO: REPLACE THIS CUSTOM LAYER WITH keras.MelSpectrogram once Keras 3 supports QAT
  def __init__(self, params: YamnetParams, **kwargs):
    super(WaveformToLogMel, self).__init__(**kwargs)

    # Compute and store the parameters needed for the STFT and Mel filterbank
    self.params = params
    self.window_length_samples = int(round(params.sample_rate * params.stft_window_seconds))
    self.hop_length_samples = int(round(params.sample_rate * params.stft_hop_seconds))
    self.fft_length = 2 ** int(math.ceil(math.log2(float(self.window_length_samples))))
    num_spectrogram_bins = self.fft_length // 2 + 1

    # Compute the Mel filterbank
    import tensorflow as tf # TODO: Implement this without tensorflow to remove dependency
    self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(self.params.mel_bands, num_spectrogram_bins, self.params.sample_rate, self.params.mel_min_hz, self.params.mel_max_hz)
  
  def get_config(self) -> dict:
    config = super(WaveformToLogMel, self).get_config()
    config.update({'params': asdict(self.params)})
    return config
  
  @classmethod
  def from_config(cls, config: dict):
    params_dict = config.pop('params')
    params = YamnetParams(**params_dict)
    return cls(params=params, **config)

  def build(self, input_shape):
    super(WaveformToLogMel, self).build(input_shape)

  def call(self, waveform):
    # Compute the STFT of the waveform
    stft = ops.abs(ops.stft(waveform, self.window_length_samples, self.hop_length_samples, self.fft_length))

    # Convert to a log Mel spectrogram
    mel_spectrogram = ops.matmul(stft, self.mel_filterbank)
    return ops.log(mel_spectrogram + self.params.log_offset)


def create_yamnet_model(params: YamnetParams, load_pretrained_weights: bool, freeze_pretrained_layers: bool) -> keras.Model:
  model = keras.Sequential(name='YAMNET', layers=[
    keras.layers.Input(shape=(int((params.patch_window_seconds + params.stft_window_seconds - params.stft_hop_seconds) * params.sample_rate),), dtype='float32'),

    WaveformToLogMel(params),
    keras.layers.Reshape((params.patch_frames, params.patch_bands, 1)),

    keras.layers.Conv2D(filters=32, kernel_size=[3, 3], strides=2, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),

    keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=1, depth_multiplier=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),
    keras.layers.Conv2D(filters=64, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),

    keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=2, depth_multiplier=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),
    keras.layers.Conv2D(filters=128, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),

    keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=1, depth_multiplier=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),
    keras.layers.Conv2D(filters=128, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),

    keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=2, depth_multiplier=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),
    keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),

    keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=1, depth_multiplier=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),
    keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),

    keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=2, depth_multiplier=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),
    keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),

    keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=1, depth_multiplier=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),
    keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),

    keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=1, depth_multiplier=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),
    keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),

    keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=1, depth_multiplier=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),
    keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),

    keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=1, depth_multiplier=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),
    keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),

    keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=1, depth_multiplier=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),
    keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),

    keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=2, depth_multiplier=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),
    keras.layers.Conv2D(filters=1024, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),

    keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=1, depth_multiplier=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),
    keras.layers.Conv2D(filters=1024, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False),
    keras.layers.BatchNormalization(center=params.batchnorm_center, scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),

    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(units=params.num_classes, use_bias=True, activation=params.classifier_activation)
  ])

  if load_pretrained_weights:
    import tensorflow_hub as hub
    layers_to_freeze = []
    pretrained_model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')
    for i, var in enumerate(pretrained_model._yamnet.trainable_variables):  # type: ignore
      if model.trainable_variables[i].shape == var.shape:
        print(f"Assigning pretrained weights for: {model.trainable_variables[i].name}")
        model.trainable_variables[i].assign(var)
        if freeze_pretrained_layers:
          for layer in model.layers:
            if layer.name == model.trainable_variables[i].name.split('/')[0]:
              layers_to_freeze.append(layer)
    for layer in layers_to_freeze:
      print(f"Freezing layer: {layer.name}")
      layer.trainable = False
  return model
