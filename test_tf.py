import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("TensorFlow Hub version:", hub.__version__)

# Quick GPU check
if tf.config.list_physical_devices('GPU'):
    print("GPU is available ✅")
else:
    print("GPU is NOT available ❌")

# Try creating a simple tensor
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
c = tf.matmul(a, b)
print("Matrix multiplication result:\n", c.numpy())
