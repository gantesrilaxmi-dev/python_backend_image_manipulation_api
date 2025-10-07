import os
import io
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
from scipy.ndimage import gaussian_filter

# ----------------------
# Flask setup
# ----------------------
app = Flask(__name__)
SAVE_FOLDER = "saved_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ----------------------
# Helper functions
# ----------------------
def validate_param(value, min_val=-100, max_val=100, default=0):
    try:
        val = float(value)
        if val < min_val: return min_val
        if val > max_val: return max_val
        return val
    except Exception:
        return default

def load_image_from_bytes(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return np.array(img).astype(np.float32)
    except Exception as e:
        print(f"Image load error: {e}")
        return None

def save_image_to_buffer(arr):
    try:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Save buffer error: {e}")
        return None

# ----------------------
# Image adjustment function
# ----------------------
def adjust_image(arr, params):
    if arr is None: return arr
    try:
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=2)
        elif arr.shape[2] < 3:
            arr = np.repeat(arr[:, :, :1], 3, axis=2)

        out = arr.copy()
        # ----------------------
        # Parse parameters
        # ----------------------
        brightness = validate_param(params.get("brightness", 0)) * 2.55
        contrast = validate_param(params.get("contrast", 0)) / 100.0
        saturation = validate_param(params.get("saturation", 0)) / 100.0
        exposure = validate_param(params.get("exposure", 0)) * 1.5
        highlights = validate_param(params.get("highlights", 0)) / 100.0
        shadows = validate_param(params.get("shadows", 0)) / 100.0
        vibrance = validate_param(params.get("vibrance", 0)) / 100.0
        temperature = validate_param(params.get("temperature", 0)) * 0.5
        fading = validate_param(params.get("fading", 0)) * 2.55
        enhance = validate_param(params.get("enhance", 0)) * 2.0
        smoothness = validate_param(params.get("smoothness", 0)) / 100.0
        ambiance = validate_param(params.get("ambiance", 0)) * 2.0
        noise = max(0.0, validate_param(params.get("noise", 0), 0, 100) * 0.3)
        color_noise = max(0.0, validate_param(params.get("color_noise", 0), 0, 100) * 0.3)
        texture = validate_param(params.get("texture", 0)) / 100.0
        clarity = validate_param(params.get("clarity", 0)) / 100.0
        dehaze = validate_param(params.get("dehaze", 0)) / 100.0
        inner_spotlight = validate_param(params.get("inner_spotlight", 0)) / 100.0
        outer_spotlight = validate_param(params.get("outer_spotlight", 0)) / 100.0
        tint = validate_param(params.get("tint", 0)) * 0.5
        vignette_amount = validate_param(params.get("vignette_amount", 0)) / 100.0
        vignette_midpoint = validate_param(params.get("vignette_midpoint", 50)) / 100.0
        vignette_feather = int(validate_param(params.get("vignette_feather", 50)))
        if vignette_feather % 2 == 0: vignette_feather += 1
        vignette_roundness = validate_param(params.get("vignette_roundness", 0)) / 100.0
        vignette_highlights = validate_param(params.get("vignette_highlights", 0)) / 100.0
        grain_amount = validate_param(params.get("grain_amount", 0)) / 100.0
        grain_size = int(validate_param(params.get("grain_size", 1), 1, 10))
        grain_roughness = validate_param(params.get("grain_roughness", 1), 1, 10)
        sharpen_amount = validate_param(params.get("sharpen_amount", 0))
        sharpen_radius = int(validate_param(params.get("sharpen_radius", 1)))
        sharpen_detail = validate_param(params.get("sharpen_detail", 1))
        sharpen_masking = validate_param(params.get("sharpen_masking", 0))
        # ----------------------
        # Brightness/Exposure/Fading
        # ----------------------
        out = out + brightness + exposure + fading
        out = np.clip(out, 0, 255)
        # Temperature (R/B)
        if temperature != 0:
            out[:, :, 0] = np.clip(out[:, :, 0] + temperature, 0, 255)
            out[:, :, 2] = np.clip(out[:, :, 2] - temperature, 0, 255)
        # Tint (G)
        if tint != 0:
            out[:, :, 1] = np.clip(out[:, :, 1] + tint, 0, 255)
        # Highlights & Shadows
        gray = np.mean(out, axis=2)
        bright_mask = (gray > 128).astype(np.float32)[..., None]
        dark_mask = (gray <= 128).astype(np.float32)[..., None]
        out += bright_mask * (highlights * 50.0)
        out += dark_mask * (shadows * 50.0)
        out = np.clip(out, 0, 255)
        # Convert to PIL for enhancements
        img = Image.fromarray(out.astype(np.uint8))
        if contrast != 0: img = ImageEnhance.Contrast(img).enhance(1.0 + contrast)
        if saturation != 0: img = ImageEnhance.Color(img).enhance(1.0 + saturation)
        if vibrance != 0:
            img_np = np.array(img).astype(np.float32)
            mean_val = np.mean(img_np, axis=2, keepdims=True)
            img_np = img_np + (img_np - mean_val) * vibrance
            img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        if enhance != 0: img = ImageEnhance.Sharpness(img).enhance(1.0 + (enhance / 50.0))
        if smoothness != 0:
            img_np = np.array(img).astype(np.float32)
            img_np = gaussian_filter(img_np, sigma=[smoothness*5, smoothness*5, 0])
            img = Image.fromarray(np.clip(img_np,0,255).astype(np.uint8))
        if ambiance != 0: img = ImageEnhance.Brightness(img).enhance(1.0 + ambiance / 50.0)
        if texture != 0: img = img.filter(ImageFilter.DETAIL if texture>0 else ImageFilter.SMOOTH)
        if clarity != 0: img = ImageEnhance.Contrast(img).enhance(1.0 + clarity)
        if dehaze != 0: img = ImageOps.autocontrast(img, cutoff=abs(int(dehaze*10)))
        # Convert back to numpy
        img_np = np.array(img).astype(np.float32)
        # Noise/Color noise
        if noise>0: img_np += np.random.normal(0, noise, img_np.shape)
        if color_noise>0: img_np += np.random.normal(0, color_noise, img_np.shape)
        # Grain
        if grain_amount>0:
            h,w=img_np.shape[:2]
            grain=np.random.normal(0,grain_amount*30,(h,w))
            k=max(1, int(grain_size*2)+1)
            grain=cv2.GaussianBlur(grain,(k,k),0)
            grain*=grain_roughness
            img_np+=np.repeat(grain[:,:,None],3,axis=2)
        # Vignette
        if vignette_amount>0:
            h,w=img_np.shape[:2]
            X,Y=np.meshgrid(np.arange(w), np.arange(h))
            cx,cy=w//2,h//2
            dist=np.sqrt((X-cx)**2 + (Y-cy)**2)
            max_dist=np.sqrt(cx**2+cy**2)
            mask = 1 - vignette_amount * np.clip((dist/max_dist - vignette_midpoint)/(1-vignette_midpoint),0,1)
            if vignette_feather>1:
                mask = cv2.GaussianBlur(mask,(vignette_feather,vignette_feather),0)
            img_np = img_np * mask[:,:,None]
        # Sharpen
        if sharpen_amount!=0:
            pil_img = Image.fromarray(np.clip(img_np,0,255).astype(np.uint8))
            factor = 1.0 + (sharpen_amount/50.0)
            pil_img = ImageEnhance.Sharpness(pil_img).enhance(factor)
            img_np = np.array(pil_img).astype(np.float32)
        return np.clip(img_np,0,255).astype(np.float32)
    except Exception as e:
        print(f"Adjustment error: {e}, arr shape: {getattr(arr,'shape',None)}, params: {params}")
        return arr

# ----------------------
# Blur functions
# ----------------------
def apply_gaussian_blur(img, strength):
    sigma = np.clip((strength / 101) * 30 + 0.1, 0.1, 50)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return blurred

def linear_blur(img, strength, strip_position=0.5, strip_width=0.3):
    h, w = img.shape[:2]
    blurred = apply_gaussian_blur(img, strength)
    mask = np.zeros((h, w), dtype=np.float32)
    strip_px = int(strip_width * max(w,h))
    start = int(strip_position * max(w,h) - strip_px // 2)
    start = np.clip(start, 0, max(w,h))
    end = np.clip(start + strip_px, 0, max(w,h))
    if w >= h:
        mask[:, start:end] = 1.0
    else:
        mask[start:end, :] = 1.0
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=strength/2, sigmaY=strength/2)[..., None]
    return (blurred * (1 - mask) + img * mask).astype(np.uint8)

def radial_blur(img, strength, radius_ratio=0.3):
    h, w = img.shape[:2]
    center = (w//2, h//2)
    radius = int(radius_ratio * min(w, h))
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, center, radius, 1, -1)
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=strength/2, sigmaY=strength/2)[..., None]
    blurred = apply_gaussian_blur(img, strength)
    return (blurred * (1 - mask) + img * mask).astype(np.uint8)

def oval_blur(img, strength, width_ratio=0.45, height_ratio=0.25):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w//2, h//2)
    axes = (int(width_ratio*w), int(height_ratio*h))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=strength/2, sigmaY=strength/2)[..., None]
    blurred = apply_gaussian_blur(img, strength)
    return (blurred * (1 - mask) + img * mask).astype(np.uint8)

def focus_blur(img, strength, focus_region='center'):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    if focus_region == 'center':
        x1, y1, x2, y2 = w//4, h//4, 3*w//4, 3*h//4
    elif focus_region == 'left':
        x1, y1, x2, y2 = 0, 0, w//2, h
    elif focus_region == 'right':
        x1, y1, x2, y2 = w//2, 0, w, h
    elif focus_region == 'top':
        x1, y1, x2, y2 = 0, 0, w, h//2
    elif focus_region == 'bottom':
        x1, y1, x2, y2 = 0, h//2, w, h
    else:
        x1, y1, x2, y2 = 0, 0, w, h
    mask[y1:y2, x1:x2] = 1.0
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=strength/2, sigmaY=strength/2)[..., None]
    blurred = apply_gaussian_blur(img, strength)
    return (blurred * (1 - mask) + img * mask).astype(np.uint8)

def hand_blur(img, strength, hand_x, hand_y, hand_radius, hand_feather):
    h, w = img.shape[:2]
    blurred = apply_gaussian_blur(img, strength)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (hand_x, hand_y), int(hand_radius), 1, -1)
    k = max(3, int(hand_feather)*2 + 1)
    mask = cv2.GaussianBlur(mask, (k,k), sigmaX=hand_feather, sigmaY=hand_feather)[..., None]
    return (blurred * (1 - mask) + img * mask).astype(np.uint8)

# ----------------------
# API Endpoints
# ----------------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message":"Image Editor API running successfully!"})

@app.route("/edit", methods=["POST"])
def edit_image():
    if "image" not in request.files:
        return jsonify({"error":"No image uploaded"}),400
    try:
        file_bytes=request.files["image"].read()
        arr=load_image_from_bytes(file_bytes)
        if arr is None: return jsonify({"error":"Failed to load image"}),400
        # sliders
        expected_keys=[
            "brightness","contrast","saturation","exposure","highlights","shadows","vibrance",
            "temperature","hue","fading","enhance","smoothness","ambiance","noise","color_noise",
            "inner_spotlight","outer_spotlight","tint","texture","clarity","dehaze","grain_amount",
            "grain_size","grain_roughness","sharpen_amount","sharpen_radius","sharpen_detail",
            "sharpen_masking","vignette_amount","vignette_midpoint","vignette_feather",
            "vignette_roundness","vignette_highlights"
        ]
        params={k:request.form.get(k,0) for k in expected_keys}
        # adjust image
        adjusted=adjust_image(arr,params)
        buf=save_image_to_buffer(adjusted)
        if buf is None: return jsonify({"error":"Failed to create image buffer"}),500
        return send_file(buf,mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/blur", methods=["POST"])
def blur_endpoint():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]

    blur_type = request.form.get('blur_type', 'linear')
    strength = int(request.form.get('blur_strength', 51))

    strip_position = float(request.form.get('strip_position', 0.5))
    strip_width = float(request.form.get('strip_width', 0.3))
    radius_ratio = float(request.form.get('radius_ratio', 0.3))
    width_ratio = float(request.form.get('width_ratio', 0.45))
    height_ratio = float(request.form.get('height_ratio', 0.25))
    focus_region = request.form.get('focus_region', 'center')

    hand_x = int(request.form.get('hand_x', w//2))
    hand_y = int(request.form.get('hand_y', h//2))
    hand_radius = float(request.form.get('hand_radius', 50))
    hand_feather = float(request.form.get('hand_feather', strength/2))

    if blur_type == 'linear':
        out = linear_blur(img, strength, strip_position, strip_width)
    elif blur_type == 'radial':
        out = radial_blur(img, strength, radius_ratio)
    elif blur_type == 'circular':
        out = radial_blur(img, strength, radius_ratio)
    elif blur_type == 'oval':
        out = oval_blur(img, strength, width_ratio, height_ratio)
    elif blur_type == 'focus':
        out = focus_blur(img, strength, focus_region)
    elif blur_type == 'hand':
        out = hand_blur(img, strength, hand_x, hand_y, hand_radius, hand_feather)
    else:
        out = apply_gaussian_blur(img, strength)

    _, buffer = cv2.imencode('.jpg', out)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/jpeg')


@app.route('/enhance', methods=['POST'])
def enhance_image():
    if "image" not in request.files:
        return jsonify({"error":"No image uploaded"}), 400
    try:
        file_bytes = request.files["image"].read()
        arr = load_image_from_bytes(file_bytes)
        if arr is None:
            return jsonify({"error":"Failed to load image"}), 400

        # Only enhancement parameters
        expected_keys = ["enhance", "clarity", "texture"]
        params = {k: request.form.get(k, 0) for k in expected_keys}

        # Apply enhancement using your adjust_image function
        enhanced = adjust_image(arr, params)
        buf = save_image_to_buffer(enhanced)
        if buf is None:
            return jsonify({"error":"Failed to save buffer"}), 500

        return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------
# Run Flask
# ----------------------
if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
