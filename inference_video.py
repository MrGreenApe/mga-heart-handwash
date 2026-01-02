#!/usr/bin/env python3
"""Run TFLite inference on a prerecorded video file."""

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from scipy import stats


CLASS_NAMES = [
    "Step 7",
    "Step 1",
    "Step 2",
    "Step 3",
    "Step 4",
    "Step 5",
    "Step 6",
]


def preprocess_frame(frame: np.ndarray, width: int, height: int, normalize: bool = True) -> np.ndarray:
    """Resize and normalize a frame to match training preprocessing."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (width, height))
    frame = frame.astype("float32")
    if normalize:
        frame = frame / 127.5 - 1.0
    return frame


def build_input_tensor(frames: deque, time_distributed: bool) -> np.ndarray:
    """Assemble frames into the tensor shape expected by the model."""
    if time_distributed:
        # frames already stored as (H, W, C)
        data = np.stack(frames, axis=0)  # (T, H, W, C)
        data = np.expand_dims(data, axis=0)  # (1, T, H, W, C)
    else:
        data = frames[-1][np.newaxis, ...]  # (1, H, W, C)
    return data.astype("float32")


class MobileNetPreprocessingLayer(tf.keras.layers.Layer):
    """Custom preprocessing layer used during training."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs / 127.5 - 1.0

    def compute_output_shape(self, input_shape):
        return input_shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TFLite inference on a video file.")
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("--model_name", default="kaggle-single-frame-final-model",
                        help="TFLite model name (without extension) located in results/models/.")
    parser.add_argument("--len_buffer", type=int, default=5,
                        help="Length of the smoothing buffer for predictions.")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply training normalization (divide by 127.5 and subtract 1).")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Process every Nth frame from the video (default: 1).")
    parser.add_argument("--display", action="store_true",
                        help="Show annotated frames in a window while processing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    model_path = Path("results/models") / f"{args.model_name}.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Keras model not found: {model_path}")

    custom_objects = {"MobileNetPreprocessingLayer": MobileNetPreprocessingLayer}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    input_shape = model.inputs[0].shape
    if len(input_shape) == 4:
        time_distributed = False
        _, img_height, img_width, _ = input_shape
        frames_required = 1
    elif len(input_shape) == 5:
        time_distributed = True
        _, frames_required, img_height, img_width, _ = input_shape
    else:
        raise ValueError(f"Unsupported input shape: {input_shape}")

    frame_buffer: deque[np.ndarray] = deque(maxlen=frames_required)
    last_predictions: deque[int] = deque(maxlen=args.len_buffer)
    last_confidences: deque[float] = deque(maxlen=args.len_buffer)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            if frame_index % args.frame_step != 0:
                continue

            processed = preprocess_frame(frame, img_width, img_height, normalize=args.normalize)
            frame_buffer.append(processed)

            if len(frame_buffer) < frames_required:
                continue  # wait until buffer is filled for time-distributed models

            input_tensor = build_input_tensor(frame_buffer, time_distributed)
            output_data = model.predict(input_tensor, verbose=0)

            predicted_class = int(np.argmax(output_data))
            confidence = float(np.max(output_data))

            last_predictions.append(predicted_class)
            last_confidences.append(confidence)

            mode_result = stats.mode(last_predictions, keepdims=False)
            mode_prediction = int(mode_result.mode) if mode_result.count > 0 else predicted_class

            avg_confidence = float(np.mean(last_confidences))
            predicted_label = CLASS_NAMES[mode_prediction] if mode_prediction < len(CLASS_NAMES) else str(mode_prediction)

            if args.display:
                overlay = frame.copy()
                text = f"{predicted_label} ({avg_confidence:.2f})"
                cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Video Inference", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if not last_predictions:
            print("No predictions generated. Check frame_step or model input expectations.")
        else:
            print(f"Final prediction: {predicted_label} with avg confidence {avg_confidence:.3f}")
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
