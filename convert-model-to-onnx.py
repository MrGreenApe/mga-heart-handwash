#!/usr/bin/env python3

"""Export a saved Keras model to ONNX format.

Mirrors convert-model-to-tflite.py behavior for generating ONNX artifacts.
"""

import argparse
import os
import sys
from typing import Optional

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Layer

try:
    import tf2onnx
except ImportError as err:
    print("Error: tf2onnx is required for ONNX conversion (pip install tf2onnx).", file=sys.stderr)
    raise

MODEL_DIR = "./results/models/"
DEFAULT_MODEL_NAME = "kaggle-videos-final-model"
DEFAULT_OPSET = 17


class MobileNetPreprocessingLayer(Layer):
    """Custom preprocessing layer used by the training pipeline."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        return preprocess_input(x)

    def compute_output_shape(self, input_shape):
        return input_shape


CUSTOM_OBJECTS = {"MobileNetPreprocessingLayer": MobileNetPreprocessingLayer}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Keras model (.keras) to ONNX format.")
    parser.add_argument("model_name", nargs="?", default=DEFAULT_MODEL_NAME,
                        help="Base name of the model file inside results/models without extension.")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET,
                        help=f"Target ONNX opset version (default: {DEFAULT_OPSET}).")
    parser.add_argument("--output", dest="output_path", default=None,
                        help="Optional explicit output path for the .onnx file.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    model_path = os.path.join(MODEL_DIR, args.model_name + ".keras")
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output_path or os.path.join(MODEL_DIR, args.model_name + ".onnx")

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)

    print(f"Converting to ONNX (opset {args.opset})...")
    input_signature = []
    for idx, keras_input in enumerate(model.inputs):
        tensor_name = keras_input.name.split(":")[0]
        spec = tf.TensorSpec(shape=keras_input.shape, dtype=keras_input.dtype, name=tensor_name or f"input_{idx}")
        input_signature.append(spec)

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        output_path=output_path,
        opset=args.opset,
    )

    print(f"ONNX model saved to: {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
