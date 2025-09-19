import tensorflow as tf

saved_model_dir = "models/ssd_mobilenet_v2_coco_savedmodel"
tflite_model_path = "models/person_detection.tflite"

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save .tflite
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved at:", tflite_model_path)


