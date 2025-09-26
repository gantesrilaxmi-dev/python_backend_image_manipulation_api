from flask import Flask, request, jsonify, send_file
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import math
import cv2
import onnxruntime as ort

app = Flask(__name__)

# -----------------------------
# Helper Functions
# -----------------------------
def load_image_file(file_storage):
    """Load and resize image from file storage"""
    try:
        img = Image.open(file_storage).convert("RGB")

        # Resize very large images to max 1280px
        MAX_SIZE = 1280
        if max(img.size) > MAX_SIZE:
            img.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)

        return img, np.array(img).astype(np.float32)
    except Exception as e:
        raise Exception(f"Failed to load image: {str(e)}")

def save_image_to_bytes(image_arr, format='JPEG', quality=85):
    """Convert image array to bytes"""
    try:
        img = Image.fromarray(np.clip(image_arr, 0, 255).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format=format, quality=quality)
        buf.seek(0)
        return buf
    except Exception as e:
        raise Exception(f"Failed to save image: {str(e)}")


def apply_vignette(arr, strength):
    """Apply vignette effect with improved visibility"""
    if strength == 0:
        return arr
    
    h, w = arr.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Create coordinate grids
    x = np.arange(w) - center_x
    y = np.arange(h) - center_y
    xx, yy = np.meshgrid(x, y)
    
    # Calculate distance from center
    distance = np.sqrt(xx**2 + yy**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    normalized_distance = distance / max_distance
    
    # Apply stronger vignette effect
    vignette_strength = abs(strength) / 100.0 * 1.5  # Increased intensity
    
    if strength > 0:
        # Dark vignette (darken edges)
        mask = 1 - (normalized_distance ** 1.5) * vignette_strength
        mask = np.clip(mask, 0.2, 1.0)  # Don't make it completely black
    else:
        # Light vignette (brighten edges) 
        mask = 1 + (normalized_distance ** 1.2) * vignette_strength
        mask = np.clip(mask, 0.5, 2.5)  # Stronger brightening
    
    mask = mask[:, :, np.newaxis]
    return arr * mask

def add_noise(arr, strength, seed=42):
    """Add noise to image"""
    if strength == 0:
        return arr
    
    rng = np.random.default_rng(seed)  # Deterministic noise
    noise_strength = abs(strength) / 100.0 * 25
    noise = rng.normal(0, noise_strength, arr.shape)
    return arr + noise

# -----------------------------
# Main Image Adjustment Function (19 parameters)
# -----------------------------
def adjust_image(pil_img, arr, params):
    """Apply all image adjustments"""
    
    result_arr = arr.copy()
    working_img = pil_img.copy()

    # Extract parameters as floats with safe defaults
    try:
        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 0.0))
        saturation = float(params.get("saturation", 0.0))
        fading = float(params.get("fading", 0.0))
        exposure = float(params.get("exposure", 0.0))
        highlights = float(params.get("highlights", 0.0))
        shadows = float(params.get("shadows", 0.0))
        vibrance = float(params.get("vibrance", 0.0))
        temperature = float(params.get("temperature", 0.0))
        hue = float(params.get("hue", 0.0))
        sharpness = float(params.get("sharpness", 0.0))
        vignette = float(params.get("vignette", 0.0))
        enhance = float(params.get("enhance", 0.0))
        dehaze = float(params.get("dehaze", 0.0))
        ambiance = float(params.get("ambiance", 0.0))
        noise = float(params.get("noise", 0.0))
        color_noise = float(params.get("colorNoise", 0.0))
        inner_spotlight = float(params.get("innerSpotlight", 0.0))
        outer_spotlight = float(params.get("outerSpotlight", 0.0))
        tint = float(params.get("tint", 0.0))

    except (ValueError, TypeError) as e:
        print(f"Parameter parsing error: {e}")
        return arr  # Return original if parsing fails

    # Apply PIL-based adjustments first
    try:
        # Brightness - Fixed range and intensity
        if brightness != 0:
            enhancer = ImageEnhance.Brightness(working_img)
            factor = 1.0 + (brightness / 100.0) * 1.5  # Increased sensitivity
            working_img = enhancer.enhance(max(0.1, min(3.0, factor)))

        # Contrast - Fixed range and intensity  
        if contrast != 0:
            enhancer = ImageEnhance.Contrast(working_img)
            factor = 1.0 + (contrast / 100.0) * 1.8  # Increased sensitivity
            working_img = enhancer.enhance(max(0.1, min(3.0, factor)))

        # Saturation - Fixed range and intensity
        if saturation != 0:
            enhancer = ImageEnhance.Color(working_img)
            factor = 1.0 + (saturation / 100.0) * 1.5  # Increased sensitivity
            working_img = enhancer.enhance(max(0.0, min(3.0, factor)))

        # Sharpness - Fixed range and intensity
        if sharpness != 0:
            enhancer = ImageEnhance.Sharpness(working_img)
            factor = 1.0 + (sharpness / 100.0) * 2.0  # Increased sensitivity
            working_img = enhancer.enhance(max(0.0, min(5.0, factor)))

        result_arr = np.array(working_img).astype(np.float32)
    except Exception as e:
        print(f"PIL enhancement error: {e}")
        result_arr = arr.copy()

    # Apply numpy-based adjustments
    try:
        # Exposure - Fixed intensity
        if exposure != 0:
            exposure_factor = 2 ** (exposure / 25.0)  # More sensitive
            result_arr *= exposure_factor

        # Highlights/Shadows - Fixed and more visible
        if highlights != 0 or shadows != 0:
            luminance = np.mean(result_arr, axis=2, keepdims=True)
            if highlights != 0:
                # Affect bright areas more strongly
                mask = np.power(np.clip((luminance - 100) / 155, 0, 1), 0.5)
                adjustment = (highlights / 100.0) * 80  # Increased effect
                result_arr += adjustment * mask
            if shadows != 0:
                # Affect dark areas more strongly  
                mask = np.power(np.clip((155 - luminance) / 155, 0, 1), 0.5)
                adjustment = (shadows / 100.0) * 80  # Increased effect
                result_arr += adjustment * mask

        # Temperature - More visible color shift
        if temperature != 0:
            temp_strength = temperature / 100.0
            result_arr[:, :, 0] += temp_strength * 50  # Red
            result_arr[:, :, 1] += temp_strength * 10  # Green (slight)
            result_arr[:, :, 2] -= temp_strength * 40  # Blue

        # Tint (shifts colors towards green/magenta)
        if tint != 0:
            # Positive values -> more green, Negative values -> more magenta
            result_arr[:, :, 0] += -tint / 100.0 * 30  # Red
            result_arr[:, :, 1] += tint / 100.0 * 30   # Green
            result_arr[:, :, 2] += -tint / 100.0 * 10  # Blue (slight)


        # Vibrance
        if vibrance != 0:
            hsv_img = Image.fromarray(np.clip(result_arr, 0, 255).astype(np.uint8)).convert('HSV')
            hsv_arr = np.array(hsv_img).astype(np.float32)
            saturation_boost = 1.0 + (vibrance / 100.0) * 1.5
            hsv_arr[:, :, 1] *= saturation_boost
            hsv_arr[:, :, 1] = np.clip(hsv_arr[:, :, 1], 0, 255)
            result_img = Image.fromarray(hsv_arr.astype(np.uint8), 'HSV').convert('RGB')
            result_arr = np.array(result_img).astype(np.float32)

        # Fading
        if fading != 0:
            fade_strength = abs(fading) / 100.0 * 0.8
            if fading > 0:
                result_arr = result_arr * (1 - fade_strength) + 255 * fade_strength
            else:
                result_arr = result_arr * (1 - fade_strength)

        # Enhance
        if enhance != 0:
            mean_val = np.mean(result_arr, axis=(0, 1), keepdims=True)
            result_arr = mean_val + (result_arr - mean_val) * (1 + enhance / 100.0 * 0.3)

        # Ambiance
        if ambiance != 0:
            luminance = np.mean(result_arr, axis=2, keepdims=True)
            mid_mask = np.exp(-((luminance - 128) ** 2) / (2 * 64 ** 2))
            result_arr += (ambiance / 100.0 * 30) * mid_mask

        # Dehaze
        if dehaze != 0:
            mean_val = np.mean(result_arr, axis=(0, 1), keepdims=True)
            result_arr = mean_val + (result_arr - mean_val) * (1 + dehaze / 100.0 * 0.3)
            gray = np.mean(result_arr, axis=2, keepdims=True)
            result_arr = gray + (result_arr - gray) * (1 + dehaze / 100.0 * 0.15)

        # Vignette
        if vignette != 0:
            result_arr = apply_vignette(result_arr, vignette)

        # Noise
        if noise != 0:
            result_arr = add_noise(result_arr, noise)

        # Color Noise
        if color_noise != 0:
            strength = abs(color_noise) / 100.0 * 15
            rng = np.random.default_rng(42)
            for c in range(3):
                channel_noise = rng.normal(0, strength, result_arr[:, :, c].shape)
                result_arr[:, :, c] += channel_noise

        # Hue shift
        if hue != 0:
            try:
                hsv = Image.fromarray(np.clip(result_arr, 0, 255).astype(np.uint8)).convert("HSV")
                hsv_arr = np.array(hsv).astype(np.float32)
                hsv_arr[:, :, 0] = (hsv_arr[:, :, 0] + (hue / 100.0) * 180) % 256
                result_arr = np.array(Image.fromarray(
                    np.clip(hsv_arr, 0, 255).astype(np.uint8), "HSV"
                ).convert("RGB")).astype(np.float32)
            except Exception as e:
                print(f"Hue adjustment error: {e}")

        # Spotlights
        if inner_spotlight != 0 or outer_spotlight != 0:
            h, w = result_arr.shape[:2]
            cx, cy = w // 2, h // 2
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / np.sqrt(cx ** 2 + cy ** 2)
            if inner_spotlight != 0:
                mask = np.exp(-(dist * 3) ** 2)
                result_arr += (inner_spotlight / 100.0 * 50) * mask[:, :, None]
            if outer_spotlight != 0:
                mask = dist
                result_arr += (outer_spotlight / 100.0 * 30) * mask[:, :, None]

    except Exception as e:
        print(f"Array processing error: {e}")

    return np.clip(result_arr, 0, 255)

# -----------------------------
# API Routes
# -----------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "Image Manipulation API is running!",
        "endpoints": {
            "/edit": "GET for info, POST with image and parameters"
        }
    })

@app.route("/edit", methods=["GET", "POST"])
def edit_route():
    if request.method == "GET":
        return jsonify({
            "message": "Image Manipulation API - Ready to process images!",
            "parameters": [
                "brightness", "contrast", "saturation", "fading", "exposure",
                "highlights", "shadows", "vibrance", "temperature", "hue",
                "sharpness", "vignette", "enhance", "dehaze", "ambiance",
                "noise", "colorNoise", "innerSpotlight", "outerSpotlight", "tint"
            ],
            "parameter_range": "All parameters accept values from -100 to 100",
            "usage": "Send POST request with 'image' file and parameter values"
        })

    # POST request handling
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # Load image
        pil_img, arr = load_image_file(request.files['image'])

        # Parse parameters as float with error handling
        params = {}
        for k in request.form.keys():
            try:
                params[k] = float(request.form.get(k, 0))
                # Clamp values to reasonable range
                params[k] = max(-100, min(100, params[k]))
            except (ValueError, TypeError):
                params[k] = 0.0

        print(f"Processing image with parameters: {params}")

        # Apply adjustments
        edited_arr = adjust_image(pil_img, arr, params)
        
        # Save result
        buf = save_image_to_bytes(edited_arr)
        return send_file(buf, mimetype="image/jpeg")

    except Exception as e:
        print(f"Processing error: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    print("Starting Image Manipulation API server...")
    print("Available at: http://localhost:5000")
    print("Supported parameters: brightness, contrast, saturation, fading, exposure,")
    print("highlights, shadows, vibrance, temperature, hue, sharpness, vignette,")
    print("enhance, dehaze, ambiance, noise, colorNoise, innerSpotlight, outerSpotlight", "tint")
    app.run(host="0.0.0.0", port=5000, debug=True)
