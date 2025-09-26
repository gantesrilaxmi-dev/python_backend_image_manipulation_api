import cv2
import numpy as np
import io
import base64
from flask import Flask, request, jsonify, send_file
from modnet import MODNet
import torch
from collections import OrderedDict

app = Flask(__name__)

# -------------------------
# Load MODNet model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modnet = MODNet(backbone_pretrained=False).to(device)
modnet.eval()

# Load checkpoint and handle possible 'state_dict' key or 'module.' prefixes
checkpoint_path = "pretrained/modnet_photographic_portrait_matting.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=device)

state_dict = checkpoint.get('state_dict', checkpoint)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.', '')  # remove 'module.' prefix if exists
    new_state_dict[name] = v

modnet.load_state_dict(new_state_dict)
print("MODNet loaded successfully!")

# -------------------------
# Load U²-Net model for general images
# -------------------------
from u2net import U2NET  # make sure u2net.py is in the same folder or installed
u2net = U2NET()
u2net.to(device)
u2net.eval()
print("U²-Net loaded successfully!")


# -------------------------
# Helper Functions
# -------------------------
def read_image(file_bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def apply_blur_to_region(image, mask, ksize=5):
    if ksize <= 0:
        return image
    kernel_size = int(ksize * 3 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 3:
        kernel_size = 3
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    mask_binary = (mask > 0.5).astype(np.float32)
    mask_3c = np.repeat(mask_binary[:, :, np.newaxis], 3, axis=2)
    result = image.astype(np.float32) * (1 - mask_3c) + blurred.astype(np.float32) * mask_3c
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_deduction_to_region(image, mask, intensity=50):
    if intensity <= 0:
        return image
    alpha = min(intensity / 100.0, 1.0)
    mask_binary = (mask > 0.5).astype(np.float32)
    mask_3c = np.repeat(mask_binary[:, :, np.newaxis], 3, axis=2)
    result = image.astype(np.float32) * (1 - mask_3c * alpha)
    return np.clip(result, 0, 255).astype(np.uint8)

def replace_background_with_color(image, background_mask, bg_color, intensity=100):
    if intensity <= 0:
        return image
    alpha = min(intensity / 100.0, 1.0)
    mask_binary = (background_mask > 0.5).astype(np.float32)
    mask_3c = np.repeat(mask_binary[:, :, np.newaxis], 3, axis=2)
    bg_layer = np.full_like(image, bg_color, dtype=np.float32)
    result = image.astype(np.float32) * (1 - mask_3c * alpha) + bg_layer * (mask_3c * alpha)
    return np.clip(result, 0, 255).astype(np.uint8)

def create_color_mask(image, target_color_bgr, tolerance=30):
    target = np.array(target_color_bgr, dtype=np.float32)
    diff = np.abs(image.astype(np.float32) - target)
    color_distance = np.sum(diff, axis=2)
    mask = (color_distance <= tolerance * 3).astype(np.float32)
    if np.sum(mask) > 100:
        mask = cv2.GaussianBlur(mask, (3,3), 0)
    return mask

def deduct_specific_color(image, color_bgr, intensity=100, tolerance=30):
    if intensity <= 0:
        return image
    color_mask = create_color_mask(image, color_bgr, tolerance)
    alpha = min(intensity / 100.0, 1.0)
    mask_3c = np.repeat(color_mask[:, :, np.newaxis], 3, axis=2)
    result = image.astype(np.float32) * (1 - mask_3c * alpha)
    return np.clip(result, 0, 255).astype(np.uint8)

def get_person_mask(image):
    original_h, original_w = image.shape[:2]
    target_h = ((original_h + 31) // 32) * 32
    target_w = ((original_w + 31) // 32) * 32
    img_resized = cv2.resize(image, (target_w, target_h))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, matte = modnet(img_tensor, True)
        matte = matte[0][0].cpu().numpy()
        matte = cv2.resize(matte, (original_w, original_h))
        matte = np.clip(matte, 0, 1)
    matte_clean = np.where(matte > 0.2, matte, 0)
    matte_clean = np.where(matte_clean > 0.6, 1.0, matte_clean)
    matte_clean = cv2.GaussianBlur(matte_clean, (5,5), 0)
    return matte_clean.astype(np.float32)

def get_dominant_colors(image, background_mask, num_colors=10):
    background_pixels = image[background_mask > 0.5]
    if len(background_pixels) == 0:
        return []
    pixels = background_pixels.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, min(num_colors, len(pixels)//10), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    colors = []
    for center in centers:
        bgr = center.astype(int)
        rgb = (bgr[2], bgr[1], bgr[0])
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        colors.append(hex_color)
    return colors

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb_color[2], rgb_color[1], rgb_color[0])

def get_salient_mask(image):
    original_h, original_w = image.shape[:2]
    img_resized = cv2.resize(image, (320, 320))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        d1, *_ = u2net(img_tensor)
        mask = d1[0,0].cpu().numpy()
        mask = cv2.resize(mask, (original_w, original_h))
        mask = np.clip(mask, 0, 1)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    return mask.astype(np.float32)


# -------------------------
# Endpoints
# -------------------------
@app.route("/analyze", methods=["POST"])
def analyze_image():
    data = request.get_json()
    if not data or "image_base64" not in data:
        return jsonify({"error": "No image_base64 provided"}), 400
    try:
        img_bytes = base64.b64decode(data["image_base64"])
        img = read_image(img_bytes)
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400
    try:
        person_mask = get_person_mask(img)
        background_mask = 1 - person_mask
        dominant_colors = get_dominant_colors(img, background_mask, 10)
        return jsonify({
            "dominant_background_colors": dominant_colors,
            "message": "These are colors found in your image's background",
            "note": "You can use ANY hex color you want for backgroundReplaceColor - not limited to these!",
            "examples": {
                "use_detected_color": "Pick one from dominant_background_colors list",
                "use_custom_color": "Use any color like #FF6B6B, #00FF00, #8A2BE2, etc.",
                "your_choice": "backgroundReplaceColor is completely YOUR decision!"
            }
        })
    except Exception as e:
        return jsonify({"error": f"Failed to analyze image: {str(e)}"}), 500

@app.route("/")
def home():
    return jsonify({
        "message": "Enhanced Edge Deduction API is running!",
        "endpoints": {
            "/analyze": "POST - Analyze image and get dominant background colors",
            "/deduction": "POST - Apply deduction effects to image",
            "/test": "GET - Test API status"
        }
    })

# -------------------------
# /deduction endpoint (robust version)
# -------------------------
@app.route("/deduction", methods=["GET", "POST"])
def api_deduction():
    if request.method == "GET":
        return jsonify({
            "message": "Send POST request with base64 image and parameters",
            "effect_targeting": {
                "PERSON_ONLY": ["foregroundDeduction", "foregroundBlur"],
                "BACKGROUND_ONLY": ["backgroundDeduction", "backgroundBlur"], 
                "ENTIRE_IMAGE": ["colorToDeduct"]
            },
            "parameters": {
                "foregroundDeduction": "0-100 (dims the PERSON only)",
                "foregroundBlur": "0-20 (blurs PERSON only)",
                "backgroundDeduction": "0-100 (affects BACKGROUND only, replaces with chosen color)",
                "backgroundBlur": "0-20 (blurs BACKGROUND only)",
                "colorToDeduct": "#RRGGBB (removes specific color from ENTIRE image)",
                "colorDeductionIntensity": "0-100 (how much to remove specific color)",
                "backgroundReplaceColor": "#RRGGBB (your chosen background color)",
                "colorTolerance": "0-100 (precision for color deduction)"
            }
        })

    data = request.get_json()
    if not data or "image_base64" not in data:
        return jsonify({"error": "No image_base64 provided"}), 400

    try:
        img_bytes = base64.b64decode(data["image_base64"])
        img = read_image(img_bytes)
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400

    # Parse parameters safely with defaults
    try:
        foreground_deduction = float(data.get("foregroundDeduction", 0))
        background_deduction = float(data.get("backgroundDeduction", 0))
        foreground_blur = float(data.get("foregroundBlur", 0))
        background_blur = float(data.get("backgroundBlur", 0))
        color_to_deduct_hex = data.get("colorToDeduct", "")
        color_deduction_intensity = float(data.get("colorDeductionIntensity", 100))
        bg_replace_hex = data.get("backgroundReplaceColor", "#000000")
        color_tolerance = float(data.get("colorTolerance", 30))
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid parameter values: {str(e)}"}), 400

    bg_replace_color = hex_to_bgr(bg_replace_hex)

    try:
        # Step 1: Get foreground mask
        person_mask = get_person_mask(img)

        # Fallback to U²-Net if MODNet fails (<5% pixels)
        if np.sum(person_mask) / (img.shape[0]*img.shape[1]) < 0.05:
            print("Portrait not detected, using U²-Net for foreground")
            person_mask = get_salient_mask(img)
            person_mask = np.where(person_mask > 0.3, 1.0, 0.0).astype(np.float32)

        # Hard masks for deduction/replacement
        foreground_mask_deduction = (person_mask > 0.5).astype(np.float32)
        background_mask_deduction = 1.0 - foreground_mask_deduction

        # Soft masks for blur effects
        foreground_mask_blur = cv2.GaussianBlur(foreground_mask_deduction, (5,5), 0) if foreground_blur>0 else foreground_mask_deduction
        background_mask_blur = cv2.GaussianBlur(background_mask_deduction, (5,5), 0) if background_blur>0 else background_mask_deduction

        # --- Initialize result ---
        result = img.astype(np.float32)

        # --- Apply PERSON effects ---
        if foreground_deduction > 0:
            result = apply_deduction_to_region(result, foreground_mask_deduction, foreground_deduction)
        if foreground_blur > 0:
            result = apply_blur_to_region(result, foreground_mask_blur, foreground_blur)

        # --- Apply BACKGROUND effects ---
        if background_deduction > 0:
            result = replace_background_with_color(result, background_mask_deduction, bg_replace_color, background_deduction)
        if background_blur > 0:
            result = apply_blur_to_region(result, background_mask_blur, background_blur)

        # --- ENTIRE IMAGE: Deduct specific color ---
        if color_to_deduct_hex and color_to_deduct_hex.startswith('#') and len(color_to_deduct_hex)==7:
            bgr_deduct = hex_to_bgr(color_to_deduct_hex)
            result = deduct_specific_color(result, bgr_deduct, color_deduction_intensity, color_tolerance)

        # --- Finalize ---
        result = np.clip(result, 0, 255).astype(np.uint8)
        success, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return jsonify({"error": "Failed to encode image"}), 500

        buffer_io = io.BytesIO(buffer)
        buffer_io.seek(0)
        return send_file(buffer_io, mimetype="image/jpeg")

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

@app.route("/test", methods=["GET"])
def test_endpoint():
    return jsonify({
        "status": "API is working",
        "model_device": str(device),
        "available_endpoints": ["/", "/analyze", "/deduction", "/test"],
        "workflow": "1. POST to /analyze to get background colors, 2. POST to /deduction with your chosen parameters"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
