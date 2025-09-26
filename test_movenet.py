import tensorflow as tf
import numpy as np
import cv2

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="movenet_singlepose_lightning.tflite")
interpreter.allocate_tensors()

# Check input details
input_details = interpreter.get_input_details()
print(input_details)
