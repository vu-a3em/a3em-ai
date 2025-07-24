from .. import keras
from keras.src.utils import file_utils

# ResNet50 model parameters
class Resnet50Params:
  num_classes: int = 1000
  use_conv_bias: bool = True
  classifier_activation: str = 'softmax'


# Helper functions to construct ResNet50 model layers
def _make_residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True):
  bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
  if conv_shortcut:
    shortcut = keras.layers.Conv2D(4 * filters, 1, strides=stride)(x)
    shortcut = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(shortcut)
  else:
    shortcut = x
  x = keras.layers.Conv2D(filters, 1, strides=stride)(x)
  x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
  x = keras.layers.ReLU()(x)
  x = keras.layers.Conv2D(filters, kernel_size, padding="same")(x)
  x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
  x = keras.layers.ReLU()(x)
  x = keras.layers.Conv2D(4 * filters, 1)(x)
  x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
  x = keras.layers.Add()([shortcut, x])
  x = keras.layers.ReLU()(x)
  return x

def _make_layer(x, filters, blocks, stride):
  x = _make_residual_block(x, filters, stride=stride)
  for _ in range(2, blocks + 1):
    x = _make_residual_block(x, filters, conv_shortcut=False)
  return x

def _make_resnet(x):
  x = _make_layer(x, 64, 3, stride=1)
  x = _make_layer(x, 128, 4, stride=2)
  x = _make_layer(x, 256, 6, stride=2)
  return _make_layer(x, 512, 3, stride=2)


# Function to create the full ResNet50 model
def create_resnet50_model(params: Resnet50Params, load_pretrained_weights: bool, freeze_pretrained_layers: bool) -> keras.Model:
  bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
  input = keras.layers.Input(shape=(224,224,3,), dtype='float32')
  x = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(input)
  x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=params.use_conv_bias)(x)
  x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
  x = keras.layers.ReLU()(x)
  x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
  x = keras.layers.MaxPooling2D(pool_size=3, strides=2)(x)
  x = _make_resnet(x)
  x = keras.layers.GlobalAveragePooling2D()(x)
  x = keras.layers.Dense(units=params.num_classes, activation=params.classifier_activation)(x)
  model = keras.Model(inputs=input, outputs=x, name='ResNet50')

  if load_pretrained_weights:
    file_name = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    file_path = 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/' + file_name
    file_hash = '2cb95161c43110f7111970584f804107'
    weights_path = file_utils.get_file(file_name, file_path, file_hash=file_hash)
    model.load_weights(weights_path, skip_mismatch=True)
    if freeze_pretrained_layers:
      for layer in model.layers:
        if not isinstance(layer, keras.layers.Dense):
          print(f"Freezing layer: {layer.name}")
          layer.trainable = False
  return model
