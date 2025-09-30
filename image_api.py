import os
import io
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import cv2
from scipy.ndimage import gaussian_filter

app = Flask(__name__)
SAVE_FOLDER = "saved_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ----------------------
# Helpers
# ----------------------
def validate_param(value, min_val=-100, max_val=100, default=0):
    try:
        val = float(value)
        return max(min_val, min(max_val, val))
    except:
        return default

def load_image_from_bytes(file_bytes):
    return np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB")).astype(np.float32)

def save_image_to_buffer(arr):
    result_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf

# ----------------------
# Image adjustments
# ----------------------
def adjust_image(arr, params):
    out = arr.copy()

    # Parse all parameters with proper validation
    brightness = validate_param(params.get("brightness", 0)) * 2.55
    contrast = validate_param(params.get("contrast", 0)) / 100
    saturation = validate_param(params.get("saturation", 0)) / 100
    exposure = validate_param(params.get("exposure", 0)) * 1.5
    highlights = validate_param(params.get("highlights", 0)) / 100
    shadows = validate_param(params.get("shadows", 0)) / 100
    vibrance = validate_param(params.get("vibrance", 0)) / 100
    temperature = validate_param(params.get("temperature", 0)) * 0.5
    hue = validate_param(params.get("hue", 0))
    fading = validate_param(params.get("fading", 0)) * 2.55
    enhance = validate_param(params.get("enhance", 0)) * 2.0
    smoothness = validate_param(params.get("smoothness", 0)) / 100
    ambiance = validate_param(params.get("ambiance", 0)) * 2.0
    noise = max(0, validate_param(params.get("noise", 0), 0, 100) * 0.3)
    color_noise = max(0, validate_param(params.get("color_noise", 0), 0, 100) * 0.3)
    inner_spotlight = validate_param(params.get("inner_spotlight", 0))
    outer_spotlight = validate_param(params.get("outer_spotlight", 0))

    # Effects
    texture = validate_param(params.get("texture", 0)) / 100
    clarity = validate_param(params.get("clarity", 0)) / 100
    dehaze = validate_param(params.get("dehaze", 0)) / 100

    # Grain
    grain_amount = validate_param(params.get("grain_amount", 0), -100, 100, 0) * 0.5
    grain_size = max(1, min(10, int(float(params.get("grain_size", 1)))))
    grain_roughness = max(1, min(10, int(float(params.get("grain_roughness", 1)))))

    # Sharpening
    sharpen_amount = int(validate_param(params.get("sharpen_amount", 0), -100, 100, 0))
    sharpen_radius = max(1, min(10, int(float(params.get("sharpen_radius", 1)))))
    sharpen_detail = max(1, min(10, int(float(params.get("sharpen_detail", 1)))))
    sharpen_masking = int(validate_param(params.get("sharpen_masking", 0), 0, 255, 0))

    # Vignette
    vignette_amount = validate_param(params.get("vignette_amount", 0), -100, 100, 0) / 100
    vignette_midpoint = validate_param(params.get("vignette_midpoint", 50), 0, 100, 50)
    vignette_feather = validate_param(params.get("vignette_feather", 50), 0, 100, 50)
    vignette_roundness = validate_param(params.get("vignette_roundness", 0), -100, 100, 0)
    vignette_highlights = validate_param(params.get("vignette_highlights", 0), 0, 100, 0) / 100

    # Apply brightness, exposure, fading
    if brightness != 0 or exposure != 0 or fading != 0:
        out = out + brightness + exposure + fading
        out = np.clip(out, 0, 255)

    # Apply temperature
    if temperature != 0:
        out[:, :, 0] = np.clip(out[:, :, 0] + temperature, 0, 255)  # Red
        out[:, :, 2] = np.clip(out[:, :, 2] - temperature, 0, 255)  # Blue

    # Apply highlights and shadows
    if highlights != 0 or shadows != 0:
        gray = np.mean(out, axis=2)
        bright_mask = (gray > 128).astype(np.float32)
        dark_mask = (gray <= 128).astype(np.float32)
        for i in range(3):
            out[:, :, i] += bright_mask * highlights * 50
            out[:, :, i] += dark_mask * shadows * 50
        out = np.clip(out, 0, 255)

    # Convert to PIL for filters
    img = Image.fromarray(out.astype(np.uint8))

    # Apply contrast
    if contrast != 0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1 + contrast)

    # Apply saturation
    if saturation != 0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1 + saturation)

    # Apply vibrance
    if vibrance != 0:
        img_np = np.array(img).astype(np.float32)
        mean_val = np.mean(img_np, axis=2, keepdims=True)
        img_np = img_np + (img_np - mean_val) * vibrance
        img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

    # Apply enhance
    if enhance != 0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1 + enhance / 50)

    # Apply smoothness (blur)
    if smoothness != 0:
        img_np = np.array(img).astype(np.float32)
        blur_amount = abs(smoothness) * 5
        img_np = gaussian_filter(img_np, sigma=[blur_amount, blur_amount, 0])
        img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

    # Apply ambiance (lightness)
    if ambiance != 0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1 + ambiance / 50)

    # Apply texture
    if texture != 0:
        if texture > 0:
            img = img.filter(ImageFilter.DETAIL)
        else:
            img = img.filter(ImageFilter.SMOOTH)

    # Apply clarity (local contrast)
    if clarity != 0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1 + clarity)

    # Apply dehaze (auto contrast)
    if dehaze != 0:
        img = ImageOps.autocontrast(img, cutoff=abs(int(dehaze * 10)))

    # Apply noise
    img_np = np.array(img).astype(np.float32)
    if noise > 0:
        noise_layer = np.random.normal(0, noise, img_np.shape)
        img_np = img_np + noise_layer
        img_np = np.clip(img_np, 0, 255)

    if color_noise > 0:
        color_noise_layer = np.random.normal(0, color_noise, img_np.shape)
        img_np = img_np + color_noise_layer
        img_np = np.clip(img_np, 0, 255)

    # Apply grain
    if grain_amount != 0:
        h, w = img_np.shape[:2]
        try:
            grain_h = max(1, h // max(1, grain_size))
            grain_w = max(1, w // max(1, grain_size))
            grain = np.random.normal(0, abs(grain_amount) * grain_roughness, (grain_h, grain_w))
            grain = cv2.resize(grain, (w, h), interpolation=cv2.INTER_LINEAR)
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] + grain
            img_np = np.clip(img_np, 0, 255)
        except Exception as e:
            print(f"Grain error: {e}")
            pass

    img = Image.fromarray(img_np.astype(np.uint8))

    # Apply sharpening
    if sharpen_amount != 0:
        try:
            amount = abs(sharpen_amount)
            if sharpen_amount > 0:
                img = img.filter(
                    ImageFilter.UnsharpMask(
                        radius=float(sharpen_radius),
                        percent=int(amount * 2),
                        threshold=int(sharpen_masking)
                    )
                )
            else:
                img = img.filter(ImageFilter.GaussianBlur(radius=float(sharpen_radius) / 2))
        except Exception as e:
            print(f"Sharpening error: {e}")
            pass

    # Apply vignette
    if vignette_amount != 0:
        try:
            np_img = np.array(img).astype(np.float32)
            rows, cols = np_img.shape[:2]
            
            # Create radial gradient mask
            center_x, center_y = cols / 2.0, rows / 2.0
            Y, X = np.ogrid[:rows, :cols]
            
            # Calculate distance from center
            dist_x = (X - center_x) / max(1, center_x)
            dist_y = (Y - center_y) / max(1, center_y)
            
            # Apply roundness
            roundness_factor = max(0.01, 1 + vignette_roundness / 100.0)
            dist = np.sqrt((dist_x ** 2) * roundness_factor + (dist_y ** 2) / roundness_factor)
            
            # Apply midpoint and feather
            midpoint = vignette_midpoint / 100.0
            feather = max(0.01, vignette_feather / 100.0)
            
            # Create vignette mask
            mask = np.clip((1.0 - dist + midpoint) / feather, 0, 1)
            
            # Apply vignette
            if vignette_amount > 0:
                # Darken edges
                vignette_strength = abs(vignette_amount)
                for i in range(3):
                    np_img[:, :, i] = np_img[:, :, i] * (1.0 - vignette_strength * (1.0 - mask))
            else:
                # Lighten edges
                vignette_strength = abs(vignette_amount)
                for i in range(3):
                    np_img[:, :, i] = np_img[:, :, i] + (255.0 - np_img[:, :, i]) * vignette_strength * (1.0 - mask)
            
            # Preserve highlights
            if vignette_highlights > 0:
                highlight_mask = (np_img > 200).astype(np.float32)
                for i in range(3):
                    np_img[:, :, i] = np_img[:, :, i] + highlight_mask[:, :, i] * vignette_highlights * 50.0
            
            img = Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))
        except Exception as e:
            print(f"Vignette error: {e}")
            pass

    # Apply spotlight effects
    if inner_spotlight != 0 or outer_spotlight != 0:
        np_img = np.array(img).astype(np.float32)
        rows, cols = np_img.shape[:2]
        Y, X = np.ogrid[:rows, :cols]
        center_y, center_x = rows / 2, cols / 2
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        spotlight_mask = 1 - (dist / max_dist)
        
        if inner_spotlight != 0:
            np_img = np_img + spotlight_mask[:, :, np.newaxis] * inner_spotlight * 2
        
        if outer_spotlight != 0:
            np_img = np_img - (1 - spotlight_mask[:, :, np.newaxis]) * outer_spotlight * 2
        
        img = Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))

    # Apply hue shift
    if hue != 0:
        img_hsv = img.convert('HSV')
        h, s, v = img_hsv.split()
        h_np = np.array(h).astype(np.int16)
        h_np = (h_np + int(hue * 2.55)) % 256
        h = Image.fromarray(h_np.astype(np.uint8))
        img = Image.merge('HSV', (h, s, v)).convert('RGB')

    # ---------------- Tint ----------------
    tint = validate_param(params.get("tint", 0), -100, 100, 0)  # -100 = green, 0 = none, +100 = magenta
    if tint != 0:
        np_img = np.array(img).astype(np.float32)
        np_img[:, :, 0] += tint    # Red channel
        np_img[:, :, 1] -= tint    # Green channel
        np_img = np.clip(np_img, 0, 255)
        img = Image.fromarray(np_img.astype(np.uint8))


    return np.array(img).astype(np.float32)

# ----------------------
# APIs
# ----------------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Image Editor API is running successfully!"})

@app.route("/edit", methods=["POST"])
def edit_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    try:
        file = request.files["image"].read()
        arr = load_image_from_bytes(file)
        
        # Parse parameters
        params = {}
        for key, value in request.form.items():
            try:
                params[key.lower()] = float(value)
            except ValueError:
                params[key.lower()] = 0
        
        # Apply adjustments
        adjusted = adjust_image(arr, params)
        
        # Return processed image
        buf = save_image_to_buffer(adjusted)
        return send_file(buf, mimetype="image/jpeg")
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Image Editor API on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", debug=True, port=5000, threaded=True)




