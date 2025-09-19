import os
from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
import io
import math

app = Flask(__name__)

# --------------------------
# Image adjustment function
# --------------------------
def adjust_image(arr, params):
    # Ensure 3 channels
    if arr.shape[2] == 4:  # RGBA
        arr = arr[:, :, :3]

    out = arr.astype(np.float32).copy()

    # Extract params safely with defaults
    brightness = params.get("brightness", 0) * 2.55
    contrast = params.get("contrast", 0)
    saturation = params.get("saturation", 0)
    fading = params.get("fading", 0) * 2.55
    exposure = params.get("exposure", 0) * 1.5
    highlights = params.get("highlights", 0)
    shadows = params.get("shadows", 0)
    vibrance = params.get("vibrance", 0)
    temperature = params.get("temperature", 0) * 0.5
    hue = params.get("hue", 0)
    sharpness = params.get("sharpness", 0)

    # ----------------------
    # Apply adjustments
    # ----------------------
    out[:, :, :3] += brightness
    out[:, :, :3] += exposure

    highlight_mask = (np.mean(out[:, :, :3], axis=2, keepdims=True) > 128)
    out[:, :, :3] += highlights * highlight_mask * 0.5

    shadow_mask = (np.mean(out[:, :, :3], axis=2, keepdims=True) <= 128)
    out[:, :, :3] += shadows * shadow_mask * 0.5

    out[:, :, :3] = out[:, :, :3] * (1 - fading / 255.0) + fading

    # Contrast
    if contrast != 0:
        factor = (259 * (contrast + 255)) / (255 * (259 - contrast + 1e-5))
        out[:, :, :3] = factor * (out[:, :, :3] - 128) + 128

    # Saturation + Vibrance
    if saturation != 0 or vibrance != 0:
        gray = np.mean(out[:, :, :3], axis=2, keepdims=True)
        sat_factor = (saturation + vibrance) / 100.0
        out[:, :, :3] = gray + (out[:, :, :3] - gray) * (1 + sat_factor)

    # Temperature
    if temperature != 0:
        out[:, :, 0] += temperature
        out[:, :, 2] -= temperature

    # Hue shift
    if hue != 0:
        angle = (hue / 100) * math.pi / 3
        u, w2 = math.cos(angle), math.sin(angle)
        r, g, b = out[:, :, 0], out[:, :, 1], out[:, :, 2]
        out[:, :, 0] = (.299 + .701 * u + .168 * w2) * r + (.587 - .587 * u + .330 * w2) * g + (.114 - .114 * u - .497 * w2) * b
        out[:, :, 1] = (.299 - .299 * u - .328 * w2) * r + (.587 + .413 * u + .035 * w2) * g + (.114 - .114 * u + .292 * w2) * b
        out[:, :, 2] = (.299 - .3 * u + 1.25 * w2) * r + (.587 - .588 * u - 1.05 * w2) * g + (.114 + .886 * u - .2 * w2) * b

    # Sharpness
    if sharpness != 0:
        sharp_factor = 1 + sharpness / 200
        out[:, :, :3] = (out[:, :, :3] - 128) * sharp_factor + 128

    # Clip to valid range
    out[:, :, :3] = np.clip(out[:, :, :3], 0, 255)

    return Image.fromarray(out.astype(np.uint8))

# --------------------------
# API Routes
# --------------------------
@app.route("/edit", methods=["GET", "POST"])
def edit_image_api():
    if request.method == "GET":
        return jsonify({"message": "✅ Send a POST request with an image and parameters to edit."})

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = Image.open(file.stream).convert("RGBA")
        arr = np.array(img)

        # Convert all parameters to int safely
        params_dict = {}
        for k, v in request.form.to_dict().items():
            try:
                params_dict[k] = int(v)
            except:
                params_dict[k] = 0

        # Adjust image
        result = adjust_image(arr, params_dict)

        # Convert to RGB before saving as JPEG
        result = result.convert("RGB")

        img_io = io.BytesIO()
        result.save(img_io, "JPEG", quality=95)
        img_io.seek(0)
        return send_file(img_io, mimetype="image/jpeg")

    except Exception as e:
        print("Error processing image:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Image Editor API is running offline!"})

# --------------------------
# Run API
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # <-- Render needs this
    app.run(debug=False, host="0.0.0.0", port=port)