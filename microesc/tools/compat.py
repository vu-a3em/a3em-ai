import torch
import numpy as np
import ai_edge_torch
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from ai_edge_torch.quantize import pt2e_quantizer, quant_config
from ai_edge_litert.interpreter import Interpreter
from torch.ao.quantization import quantize_pt2e
from typing import Literal
from microesc import keras
from .. import PyDataset

def convert_keras_to_tflite(model: keras.Model, output_file: str, quantize: bool):
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  if quantize:
    converter.optimizations.add(tf.lite.Optimize.DEFAULT)
  tflite_model = converter.convert()
  with open(output_file, 'wb') as f:
      f.write(tflite_model)  # type: ignore

def convert_tensorflow_to_tflite(saved_model_dir: str, output_file: str, quantize: bool):
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  if quantize:
    converter.optimizations.add(tf.lite.Optimize.DEFAULT)  # type: ignore
  tflite_model = converter.convert()
  with open(output_file, 'wb') as f:
      f.write(tflite_model)  # type: ignore

def convert_pytorch_to_tflite(model: torch.nn.Module, test_dataset: PyDataset, output_file: str, quantize: Literal['p2te', 'tf'] | None, input_shape: tuple | None = None):
  input_shape = input_shape if input_shape else next(model.parameters()).size()
  quant_cfg, quant_flags = None, None
  if quantize == 'pt2e':
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=True))
    model = torch.export.export_for_training(model.eval(), (torch.randn(input_shape),)).module()
    model = quantize_pt2e.prepare_pt2e(model, quantizer)  # type: ignore
    for i in range(len(test_dataset)):
      for data in test_dataset[i][0]:
        model(data)
    model = quantize_pt2e.convert_pt2e(model, fold_quantize=False)
    quant_cfg = quant_config.QuantConfig(pt2e_quantizer=quantizer)
  elif quantize == 'tf':
    quant_flags = {'optimizations': [tf.lite.Optimize.DEFAULT]}
  else:
    model = model.eval()
  tflite_model = ai_edge_torch.convert(model, (torch.randn(input_shape),), quant_config=quant_cfg, _ai_edge_converter_flags=quant_flags)
  tflite_model.export(output_file)

def load_keras_model(model_path: str) -> keras.Model:
  with tfmot.quantization.keras.quantize_scope():
    return keras.models.load_model(model_path)  # type: ignore

def load_tflite_model(model_path: str) -> Interpreter:
  interpreter = Interpreter(model_path)
  interpreter.allocate_tensors()
  return interpreter

def test_tflite_model(model_path: str, test_dataset: PyDataset) -> float:
  interpreter = load_tflite_model(model_path)
  correct_predictions, total_predictions = 0, 0
  input_index = interpreter.get_input_details()[0]['index']
  output = interpreter.tensor(interpreter.get_output_details()[0]['index'])
  for i in range(len(test_dataset)):
    batch = test_dataset[i]
    for j, data in enumerate(batch[0]):
      data = np.expand_dims(data, axis=0)
      interpreter.set_tensor(input_index, data)
      interpreter.invoke()
      prediction = np.argmax(output()[0])
      if prediction == batch[1][j]:
        correct_predictions += 1
      total_predictions += 1
  return correct_predictions / total_predictions

def run_tflite_inference(interpreter: Interpreter, input_data: np.ndarray) -> np.intp:
  input_index = interpreter.get_input_details()[0]['index']
  output = interpreter.tensor(interpreter.get_output_details()[0]['index'])
  interpreter.set_tensor(input_index, input_data)
  interpreter.invoke()
  return np.argmax(output()[0])
