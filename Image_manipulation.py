import sys
import numpy as np
from PIL import Image, ImageFilter
import math
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QTabWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

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
# Adjust Image
# --------------------------
def adjust_image(arr, params):
    h, w, _ = arr.shape
    out = arr.copy()

    # --------------------------
    # BASIC SLIDERS
    # --------------------------
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
    vignette = params.get("vignette", 0)
    enhance = params.get("enhance", 0)
    dehaze = params.get("dehaze", 0)
    ambiance = params.get("ambiance", 0)
    noise = max(0, params.get("noise", 0) * 0.3)
    color_noise = max(0, params.get("colorNoise", 0) * 0.3)
    inner_spotlight = params.get("innerSpotlight", 0)
    outer_spotlight = params.get("outerSpotlight", 0)

    out[:, :, :3] += brightness + exposure

    # Highlights / Shadows
    highlight_mask = (np.mean(out[:, :, :3], axis=2, keepdims=True) > 128)
    shadow_mask = (np.mean(out[:, :, :3], axis=2, keepdims=True) <= 128)
    out[:, :, :3] += highlights * highlight_mask * 0.5
    out[:, :, :3] += shadows * shadow_mask * 0.5

    # Fading
    out[:, :, :3] = out[:, :, :3] * (1 - fading / 255.0) + fading

    # Contrast
    if contrast != 0:
        factor = (259 * (contrast + 255)) / (255 * (259 - contrast + 1e-5))
        out[:, :, :3] = factor * (out[:, :, :3] - 128) + 128

    # Saturation / Vibrance
    if saturation != 0 or vibrance != 0:
        gray = np.mean(out[:, :, :3], axis=2, keepdims=True)
        sat_factor = (saturation + vibrance) / 100.0
        out[:, :, :3] = gray + (out[:, :, :3] - gray) * (1 + sat_factor)

    # Temperature & Hue
    out[:, :, 0] += temperature
    out[:, :, 2] -= temperature
    if hue != 0:
        angle = (hue / 100) * math.pi / 3
        u, w2 = math.cos(angle), math.sin(angle)
        r, g, b = out[:, :, 0], out[:, :, 1], out[:, :, 2]
        out[:, :, 0] = (.299 + .701 * u + .168 * w2) * r + (.587 - .587 * u + .330 * w2) * g + (.114 - .114 * u - .497 * w2) * b
        out[:, :, 1] = (.299 - .299 * u - .328 * w2) * r + (.587 + .413 * u + .035 * w2) * g + (.114 - .114 * u + .292 * w2) * b
        out[:, :, 2] = (.299 - .3 * u + 1.25 * w2) * r + (.587 - .588 * u - 1.05 * w2) * g + (.114 + .886 * w2 - 0.2 * w2) * b

    # --------------------------
    # EFFECTS
    # --------------------------
    texture = params.get("texture", 0)
    clarity = params.get("clarity", 0)
    if texture != 0 or clarity != 0:
        factor = 1 + (texture + clarity) / 100
        out[:, :, :3] = (out[:, :, :3] - 128) * factor + 128

    dehaze_effect = params.get("dehaze_effect", 0)
    if dehaze_effect != 0:
        out[:, :, :3] += dehaze_effect

    # --------------------------
    # GRAIN
    # --------------------------
    grain_amount = params.get("grain_amount", 0)
    grain_size = max(1, params.get("grain_size", 1))
    grain_roughness = params.get("grain_roughness", 0)
    grain_roundness = max(0.01, params.get("grain_roundness", 1.0))

    if grain_amount != 0:
        noise_arr = np.random.normal(0, abs(grain_amount)/2, size=(h//grain_size, w//grain_size, 3))
        noise_arr = np.array(Image.fromarray(noise_arr.astype(np.float32)).resize((w,h)))
        if grain_amount < 0:
            noise_arr *= -1
        out[:, :, :3] += noise_arr
        if grain_roughness != 0:
            out[:, :, :3] += np.random.normal(0, abs(grain_roughness), (h, w, 3))

    # --------------------------
    # SHARPENING
    # --------------------------
    sharpen_amount = params.get("sharpen_amount", 0)
    sharpen_radius = max(0.1, params.get("sharpen_radius", 1))
    if sharpen_amount != 0:
        pil_img = Image.fromarray(np.clip(out,0,255).astype(np.uint8))
        pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=sharpen_radius, percent=sharpen_amount, threshold=0))
        out = np.array(pil_img).astype(np.float32)

    # --------------------------
    # VIGNETTE
    # --------------------------
    vignette_amount = params.get("vignette_amount", 0)
    vignette_midpoint = max(0.01, params.get("vignette_midpoint", 0.5))
    vignette_feather = max(0.01, params.get("vignette_feather", 0.5))
    vignette_roundness = max(0.01, params.get("vignette_roundness", 1.0))
    vignette_highlights = params.get("vignette_highlights", 0)

    if vignette_amount != 0:
        y, x = np.indices((h, w))
        dx = (x - w/2) / (w/2) / vignette_roundness
        dy = (y - h/2) / (h/2) / vignette_roundness
        dist = np.sqrt(dx**2 + dy**2)
        mask = 1 - vignette_amount/100 * (dist ** (1/vignette_feather))
        mask = np.clip(mask + vignette_highlights/100, 0, 1)
        mask = np.nan_to_num(mask, nan=1.0)
        out[:, :, :3] *= mask[..., None]

    # --------------------------
    # NOISE
    # --------------------------
    if noise > 0:
        out[:, :, :3] += np.random.normal(0, noise, (h, w, 1))
    if color_noise > 0:
        out[:, :, :3] += np.random.normal(0, color_noise, (h, w, 3))

    # --------------------------
    # Final clip
    # --------------------------
    out[:, :, :3] = np.nan_to_num(out[:, :, :3], nan=0.0)
    out[:, :, :3] = np.clip(out[:, :, :3], 0, 255)
    result_img = Image.fromarray(out.astype(np.uint8))

    if result_img.mode == 'RGBA':
        rgb_img = Image.new('RGB', result_img.size, (255, 255, 255))
        rgb_img.paste(result_img, mask=result_img.split()[3])
        return rgb_img

    return result_img

# --------------------------
# GUI
# --------------------------
class ImageEditorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎨 Image Editor GUI")
        self.resize(1200, 800)

        self.params = {key: 0 for key in [
            'brightness','contrast','saturation','fading','exposure','highlights','shadows','vibrance',
            'temperature','hue','sharpness','vignette','enhance','dehaze','ambiance','noise','colorNoise','innerSpotlight','outerSpotlight',
            'texture','clarity','dehaze_effect',
            'grain_amount','grain_size','grain_roughness','grain_roundness',
            'sharpen_amount','sharpen_radius','sharpen_detail','sharpen_masking',
            'vignette_amount','vignette_midpoint','vignette_feather','vignette_roundness','vignette_highlights'
        ]}

        self.original_arr = load_image("input.jpg")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Image display
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        self.update_preview()

        # Tabs
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # --- Create Tabs ---
        self.create_tab("Basic", [
            'brightness','contrast','saturation','exposure','highlights','shadows','vibrance',
            'temperature','hue','sharpness','fading','vignette','enhance','dehaze','ambiance',
            'noise','colorNoise','innerSpotlight','outerSpotlight'
        ])
        self.create_tab("Effects", ['texture','clarity','dehaze_effect'])
        self.create_tab("Grain", ['grain_amount','grain_size','grain_roughness','grain_roundness'])
        self.create_tab("Sharpening", ['sharpen_amount','sharpen_radius','sharpen_detail','sharpen_masking'])
        self.create_tab("Vignette", ['vignette_amount','vignette_midpoint','vignette_feather','vignette_roundness','vignette_highlights'])

        # Save button
        save_btn = QPushButton("Save Image")
        save_btn.clicked.connect(self.save_image)
        self.layout.addWidget(save_btn)

    def create_tab(self, name, sliders):
        tab = QWidget()
        vbox = QVBoxLayout()
        tab.setLayout(vbox)

        for s in sliders:
            hbox = QHBoxLayout()
            label = QLabel(f"{s}: 0")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(-100)
            slider.setMaximum(100)
            slider.setValue(0)
            slider.valueChanged.connect(lambda value, key=s, l=label: self.slider_changed(key, value, l))
            hbox.addWidget(label)
            hbox.addWidget(slider)
            vbox.addLayout(hbox)

        self.tabs.addTab(tab, name)

    def slider_changed(self, key, value, label):
        label.setText(f"{key}: {value}")
        self.params[key] = value
        self.update_preview()

    def update_preview(self):
        preview = adjust_image(self.original_arr.copy(), self.params)
        preview = preview.convert("RGB")
        data = preview.tobytes("raw", "RGB")
        qimage = QImage(data, preview.width, preview.height, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def save_image(self):
        final = adjust_image(self.original_arr.copy(), self.params)
        final.save("output.jpg")
        print("✅ Image saved as output.jpg")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ImageEditorGUI()
    gui.show()
    sys.exit(app.exec())
