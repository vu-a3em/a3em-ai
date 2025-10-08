from microesc.classification.Yamnet import YamnetParams, WaveformToLogMel
from microesc import keras

def build_mini_yamnet_model(params: YamnetParams, blocks , dense_units) -> keras.Model:
  # A smaller version of Yamnet for less capable hardware
  def _build_block(filters: int, kernel_size: list, strides: int) -> list[keras.layers.Layer]:
    return [
      keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, depth_multiplier=1, padding=params.conv_padding, use_bias=False),
      keras.layers.GroupNormalization(groups=8, axis=-1, epsilon=params.batchnorm_epsilon),
      keras.layers.ReLU(max_value=6.0),
      keras.layers.Conv2D(filters=filters, kernel_size=[1, 1], strides=1, padding=params.conv_padding, use_bias=False),
      keras.layers.GroupNormalization(groups=8, axis=-1, epsilon=params.batchnorm_epsilon),
      keras.layers.ReLU(max_value=6.0),
    ]

  layers = [
      keras.layers.Input(shape=(int((params.patch_window_seconds + params.stft_window_seconds - params.stft_hop_seconds) * params.sample_rate),), dtype='float32'),

    WaveformToLogMel(params),
    keras.layers.Reshape((params.patch_frames, params.patch_bands, 1)),

    keras.layers.Conv2D(filters=32, kernel_size=[3, 3], strides=2, padding=params.conv_padding, use_bias=False),
    keras.layers.GroupNormalization(groups=8, axis=-1, epsilon=params.batchnorm_epsilon),
    keras.layers.ReLU(max_value=6.0),
  ]

  for (filters, kernel_size, strides) in blocks:
    layers += _build_block(filters, kernel_size, strides)

  # Pooling after the conv layers
  layers.append(
    keras.layers.GlobalAveragePooling2D(),
  )

  # Fully connected layers
  for units in dense_units:
    layers += [
      keras.layers.Dense(units=units, activation='relu', use_bias=False,
                         kernel_regularizer=keras.regularizers.l2(1e-4)),
      keras.layers.BatchNormalization(center=True, scale=True, epsilon=params.batchnorm_epsilon),
      keras.layers.Dropout(0.3),
    ]

  # Classifier layer
  layers.append(
    keras.layers.Dense(units=params.num_classes, use_bias=True, activation=params.classifier_activation)
  )

  model = keras.Sequential(name='YAMNET-MINI', layers=layers)
  return model
