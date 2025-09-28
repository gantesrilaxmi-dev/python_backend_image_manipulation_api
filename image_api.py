import os
import io
import uuid
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageEnhance, ImageFilter

app = Flask(__name__)

# ----------------------
# Helpers
# ----------------------
def validate_param(value, min_val=-100, max_val=100):
    """Clamp the parameter value within min and max."""
    return max(min_val, min(max_val, value))

def load_image_from_bytes(file_bytes):
    return np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB")).astype(np.float32)

def save_image(arr):
    result_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

# ----------------------
# Adjustments
# ----------------------
def adjust_image(arr, params):
    """Apply all adjustments to the image array"""
    out = arr.copy()

    # Base adjustments
    brightness = validate_param(params.get("brightness", 0)) * 2.55
    contrast = validate_param(params.get("contrast", 0)) / 100.0
    saturation = validate_param(params.get("saturation", 0)) / 100.0
    exposure = validate_param(params.get("exposure", 0)) * 1.5
    highlights = validate_param(params.get("highlights", 0)) / 100.0
    shadows = validate_param(params.get("shadows", 0)) / 100.0
    vibrance = validate_param(params.get("vibrance", 0)) / 100.0
    temperature = validate_param(params.get("temperature", 0)) * 0.5
    hue = validate_param(params.get("hue", 0))
    fading = validate_param(params.get("fading", 0)) * 2.55
    enhance = validate_param(params.get("enhance", 0)) * 2.0
    smoothness = validate_param(params.get("smoothness", 0)) / 100.0
    ambiance = validate_param(params.get("ambiance", 0)) * 2.0
    noise = max(0, validate_param(params.get("noise", 0)) * 0.3)
    color_noise = max(0, validate_param(params.get("color_noise", 0)) * 0.3)
    inner_spotlight = validate_param(params.get("inner_spotlight", 0))
    outer_spotlight = validate_param(params.get("outer_spotlight", 0))

    # Effects
    texture = validate_param(params.get("texture", 0))
    clarity = validate_param(params.get("clarity", 0))
    dehaze = validate_param(params.get("dehaze", 0))

    # Grain
    grain_amount = validate_param(params.get("grain_amount", 0))
    grain_size = max(1, params.get("grain_size", 1))
    grain_roughness = max(1, params.get("grain_roughness", 1))

    # Sharpening
    sharpen_amount = validate_param(params.get("sharpen_amount", 0))
    sharpen_radius = max(1, params.get("sharpen_radius", 1))
    sharpen_detail = max(1, params.get("sharpen_detail", 1))
    sharpen_masking = validate_param(params.get("sharpen_masking", 0))

    # Vignette
    vignette_amount = validate_param(params.get("vignette_amount", 0))
    vignette_midpoint = validate_param(params.get("vignette_midpoint", 50), 0, 100)
    vignette_feather = validate_param(params.get("vignette_feather", 50), 0, 100)
    vignette_roundness = validate_param(params.get("vignette_roundness", 0), -100, 100)
    vignette_highlights = validate_param(params.get("vignette_highlights", 0), 0, 100)

    # ---- Apply simple effects (brightness/exposure/fading/contrast) ----
    out[:, :, :3] += brightness + exposure + fading
    out = np.clip(out, 0, 255)

    # For more advanced effects, you can integrate PIL ImageEnhance
    img = Image.fromarray(out.astype(np.uint8))
    if contrast != 0:
        img = ImageEnhance.Contrast(img).enhance(1 + contrast)
    if saturation != 0:
        img = ImageEnhance.Color(img).enhance(1 + saturation)
    if sharpen_amount != 0:
        img = img.filter(ImageFilter.UnsharpMask(radius=sharpen_radius, percent=sharpen_amount, threshold=sharpen_masking))

    out = np.array(img).astype(np.float32)
    return out

# ----------------------
# APIs
# ----------------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "🎨 Image Manipulation API is running!"})

@app.route("/edit", methods=["POST"])
def edit_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files["image"].read()
        arr = load_image_from_bytes(file)

        params = request.form.to_dict()
        # Convert all values to float safely
        params = {k.lower(): float(v) for k, v in params.items()}

        adjusted = adjust_image(arr, params)
        buf = save_image(adjusted)
        return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
