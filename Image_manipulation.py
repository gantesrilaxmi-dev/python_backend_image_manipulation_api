import sys
import numpy as np
from PIL import Image
import math
import os
import platform

# --------------------------
# Load Image
# --------------------------
def load_image(path):
    try:
        img = Image.open(path).convert("RGBA")
        return np.array(img).astype(np.float32)
    except:
        print("❌ input.jpg not found!")
        sys.exit(1)

# --------------------------
# Open image in default viewer
# --------------------------
def open_image(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":  # macOS
        os.system(f"open '{path}'")
    else:  # Linux
        os.system(f"xdg-open '{path}'")

# --------------------------
# Parameter validation
# --------------------------
def validate_param(key, value):
    """Validate parameter ranges: fading 0-100, others -100 to 100"""
    if key == "fading":
        return max(0, min(100, value))
    else:
        return max(-100, min(100, value))

# --------------------------
# Adjustments
# --------------------------
def adjust_image(arr, params):
    h, w, _ = arr.shape
    out = arr.copy()

    # Extract params (normalized to appropriate ranges)
    brightness = params["brightness"] * 2.55
    contrast = params["contrast"]
    saturation = params["saturation"]
    fading = params["fading"] * 2.55
    exposure = params["exposure"] * 1.5
    highlights = params["highlights"] * 1.0
    shadows = params["shadows"] * 1.0
    vibrance = params["vibrance"]
    temperature = params["temperature"] * 0.5
    hue = params["hue"]
    sharpness = params["sharpness"]
    vignette = params["vignette"]
    enhance = params["enhance"] * 2.0
    dehaze = params["dehaze"]
    ambiance = params["ambiance"] * 2.0
    noise = max(0, params["noise"] * 0.3)
    color_noise = max(0, params["colorNoise"] * 0.3)
    inner_spotlight = params["innerSpotlight"]
    outer_spotlight = params["outerSpotlight"]

    # Basic adjustments
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

    # Vignette and Spotlights
    if vignette != 0 or inner_spotlight != 0 or outer_spotlight != 0:
        y, x = np.indices((h, w))
        dx, dy = x - w / 2, y - h / 2
        dist = np.sqrt(dx ** 2 + dy ** 2) / (np.sqrt(w ** 2 + h ** 2) / 2)
        dist = np.clip(dist, 0, 1)
        vign = (1 - vignette / 100 * dist) ** 2 if vignette > 0 else (1 + abs(vignette) / 100 * (1 - dist)) ** 0.5
        inner_factor = (1 + inner_spotlight / 100 * (1 - dist)) ** 1.2 if inner_spotlight != 0 else 1
        outer_factor = (1 + outer_spotlight / 100 * dist) ** 1.2 if outer_spotlight != 0 else 1
        vign_mask = vign * inner_factor * outer_factor
        out[:, :, :3] *= vign_mask[..., None]

    # Enhance + Dehaze
    if enhance != 0 or dehaze != 0:
        deh_factor = 1 + dehaze / 200
        out[:, :, :3] = (out[:, :, :3] + enhance - 128) * deh_factor + 128

    # Ambiance
    if ambiance != 0:
        out[:, :, 0] += ambiance / 4
        out[:, :, 1] += ambiance / 6
        out[:, :, 2] += ambiance / 5

    # Noise
    if noise > 0 or color_noise > 0:
        if noise > 0:
            gray_noise = np.random.normal(0, noise, size=(h, w, 1))
            out[:, :, :3] += gray_noise
        if color_noise > 0:
            color_noise_arr = np.random.normal(0, color_noise, size=(h, w, 3))
            out[:, :, :3] += color_noise_arr

    out[:, :, :3] = np.clip(out[:, :, :3], 0, 255)

    result_img = Image.fromarray(out.astype(np.uint8))
    if result_img.mode == 'RGBA':
        rgb_img = Image.new('RGB', result_img.size, (255, 255, 255))
        rgb_img.paste(result_img, mask=result_img.split()[3] if len(result_img.split()) == 4 else None)
        return rgb_img
    return result_img

# --------------------------
# Display parameters
# --------------------------
def display_params(params):
    print("\n📊 Current Parameters:")
    print("=" * 40)
    for key, value in params.items():
        range_info = "(0 to 100)" if key == "fading" else "(-100 to 100)"
        print(f"{key:15}: {value:4} {range_info}")
    print("=" * 40)

# --------------------------
# Main Interactive Loop
# --------------------------
def main():
    params = {key: 0 for key in [
        'brightness','contrast','saturation','fading','exposure',
        'highlights','shadows','vibrance','temperature','hue',
        'sharpness','vignette','enhance','dehaze',
        'ambiance','noise','colorNoise','innerSpotlight','outerSpotlight'
    ]}

    print("🎨 Interactive Image Editor")
    print("=" * 50)
    print("Commands: set <param> <value>, show, reset, save, help, quit")
    print("Preview now updates automatically after each set command!\n")

    original_arr = load_image("input.jpg")
    print("✅ input.jpg loaded successfully!")

    while True:
        print("\n> ", end="")
        cmd = input().strip().lower()

        if cmd in ["quit","exit","q"]:
            print("👋 Goodbye!")
            break

        elif cmd == "save":
            final = adjust_image(original_arr.copy(), params)
            final.save("output.jpg")
            print("✅ Final image saved as 'output.jpg'")
            open_image("output.jpg")

        elif cmd in ["show","display","params"]:
            display_params(params)

        elif cmd == "reset":
            for key in params: params[key] = 0
            print("🔄 All parameters reset to 0")
            display_params(params)

        elif cmd.startswith("set "):
            parts = cmd.split()
            if len(parts) != 3:
                print("❌ Usage: set <parameter> <value>")
                continue

            param_name, value_str = parts[1], parts[2]
            if param_name not in params:
                print(f"❌ Unknown parameter: {param_name}")
                continue

            try:
                value = int(value_str)
                validated_value = validate_param(param_name, value)
                params[param_name] = validated_value
                print(f"✅ {param_name} set to {validated_value}")

                # Automatic preview
                preview = adjust_image(original_arr.copy(), params)
                preview.save("preview.jpg")
                print("👁️  Preview updated")
                open_image("preview.jpg")

            except ValueError:
                print("❌ Value must be an integer")

        elif cmd in ["help","h"]:
            print("Commands: set <param> <value>, show, reset, save, help, quit")

        else:
            if cmd:
                print(f"❌ Unknown command: {cmd}")

if __name__ == "__main__":
    main()





