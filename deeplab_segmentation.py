import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load a valid DeepLabV3 model with MobileNetV2 backbone
MODEL_URL = "https://tfhub.dev/tensorflow/deeplabv3_mobilenetv2/1"
model = hub.load(MODEL_URL)
print("Model loaded successfully!")

# Function to preprocess the image
def preprocess_image(image_path, target_size=(513, 513)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]
    img_resized = cv2.resize(img, target_size)
    img_tensor = tf.convert_to_tensor(img_resized, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, 0)  # Add batch dimension
    return img_tensor, original_size

# Function to run segmentation
def run_segmentation(image_path):
    img_tensor, original_size = preprocess_image(image_path)
    result = model(img_tensor)
    seg_map = tf.argmax(result['default'], axis=-1)
    seg_map = tf.squeeze(seg_map).numpy().astype(np.uint8)

    # Resize segmentation mask back to original image size
    seg_map = cv2.resize(seg_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return seg_map

# Function to overlay mask on image
def overlay_mask(image_path, seg_map):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a color map for visualization
    colormap = np.array([
        [0, 0, 0],       # background
        [128, 0, 0],     # aeroplane
        [0, 128, 0],     # bicycle
        [128, 128, 0],   # bird
        [0, 0, 128],     # boat
        [128, 0, 128],   # bottle
        [0, 128, 128],   # bus
        [128, 128, 128], # car
        [64, 0, 0],      # cat
        [192, 0, 0],     # chair
        # add more if needed for other classes
    ], dtype=np.uint8)

    # Apply color map
    mask_rgb = colormap[seg_map % len(colormap)]
    overlay = cv2.addWeighted(img, 0.7, mask_rgb, 0.3, 0)
    
    # Display result
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    image_path = "test_image.jpg"  # replace with your image path
    seg_map = run_segmentation(image_path)
    overlay_mask(image_path, seg_map)

