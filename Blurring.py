import cv2
import numpy as np
import sys

# -------------------- BLUR FUNCTIONS --------------------

def show_image(img, title="Preview"):
    cv2.imshow(title, img)
    print("Press any key on the image window to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow(title)

def linear_blur(img, ksize=(61, 61), direction='vertical', angle=0, strip_width=0.3, position=0.5):
    h, w = img.shape[:2]
    blurred = cv2.GaussianBlur(img, ksize, 0)
    mask = np.zeros((h, w), dtype=np.uint8)

    if direction == 'vertical':
        strip_w = int(w * strip_width)
        strip_center = int(w * position)
        left = max(0, strip_center - strip_w // 2)
        right = min(w, strip_center + strip_w // 2)
        cv2.rectangle(mask, (left, 0), (right, h), 255, -1)
    elif direction == 'horizontal':
        strip_h = int(h * strip_width)
        strip_center = int(h * position)
        top = max(0, strip_center - strip_h // 2)
        bottom = min(h, strip_center + strip_h // 2)
        cv2.rectangle(mask, (0, top), (w, bottom), 255, -1)
    elif direction == 'angle':
        strip_w = int(w * strip_width)
        strip_center = int(w * position)
        left = max(0, strip_center - strip_w // 2)
        right = min(w, strip_center + strip_w // 2)
        cv2.rectangle(mask, (left, 0), (right, h), 255, -1)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        mask = cv2.warpAffine(mask, M, (w, h))

    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    mask_3ch = mask[..., None] / 255.0
    result = (img * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
    show_image(result, "Linear Blur Preview")
    return result

def radial_blur(img, ksize=(71, 71), center=None, inner_radius_ratio=0.15, outer_radius_ratio=0.6):
    h, w = img.shape[:2]
    blurred = cv2.GaussianBlur(img, ksize, 0)
    if center is None:
        center = (w // 2, h // 2)
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    max_dist = np.sqrt((w/2)**2 + (h/2)**2)
    dist_normalized = dist_from_center / max_dist
    mask = np.zeros((h, w), dtype=np.float32)
    mask[dist_normalized <= inner_radius_ratio] = 1.0
    gradient_region = (dist_normalized > inner_radius_ratio) & (dist_normalized < outer_radius_ratio)
    mask[gradient_region] = 1.0 - ((dist_normalized[gradient_region] - inner_radius_ratio) / (outer_radius_ratio - inner_radius_ratio))
    mask[dist_normalized >= outer_radius_ratio] = 0.0
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask_3ch = mask[..., None]
    result = (img * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
    show_image(result, "Radial Blur Preview")
    return result

def circular_blur(img, ksize=(61, 61), radius_ratio=0.3, center=None, feather=51):
    h, w = img.shape[:2]
    blurred = cv2.GaussianBlur(img, ksize, 0)
    if center is None:
        center = (w // 2, h // 2)
    radius = int(min(h, w) * radius_ratio)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    if feather > 1:
        if feather % 2 == 0: feather += 1
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    mask_3ch = mask[..., None] / 255.0
    result = (img * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
    show_image(result, "Circular Blur Preview")
    return result

def oval_blur(img, ksize=(61, 61), width_ratio=0.45, height_ratio=0.25, center=None, feather=51):
    h, w = img.shape[:2]
    blurred = cv2.GaussianBlur(img, ksize, 0)
    if center is None:
        center = (w // 2, h // 2)
    axis_w = int(w * width_ratio)
    axis_h = int(h * height_ratio)
    axes = (axis_w, axis_h)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    if feather > 1:
        if feather % 2 == 0: feather += 1
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    mask_3ch = mask[..., None] / 255.0
    result = (img * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
    show_image(result, "Oval Blur Preview")
    return result

def focus_blur(img, ksize=(61, 61), focus_region='center', blur_outside=True):
    h, w = img.shape[:2]
    blurred = cv2.GaussianBlur(img, ksize, 0)
    mask = np.zeros((h, w), dtype=np.uint8)
    if focus_region == 'center':
        cv2.circle(mask, (w//2, h//2), min(h, w)//3, 255, -1)
    elif focus_region == 'top':
        cv2.rectangle(mask, (0, 0), (w, h//3), 255, -1)
    elif focus_region == 'bottom':
        cv2.rectangle(mask, (0, 2*h//3), (w, h), 255, -1)
    elif focus_region == 'left':
        cv2.rectangle(mask, (0, 0), (w//3, h), 255, -1)
    elif focus_region == 'right':
        cv2.rectangle(mask, (2*w//3, 0), (w, h), 255, -1)
    if not blur_outside:
        mask = 255 - mask
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    mask_3ch = mask[..., None] / 255.0
    result = (img * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
    show_image(result, "Focus Blur Preview")
    return result

# -------------------- MAIN PROGRAM --------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Blurring.py <input_image>")
        sys.exit(1)

    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Could not load image: {path}")
        sys.exit(1)

    effects = {
        "1": "Linear Blur - Vertical",
        "2": "Linear Blur - Horizontal",
        "3": "Linear Blur - Angle",
        "4": "Radial Blur",
        "5": "Circular Blur",
        "6": "Oval Blur",
        "7": "Focus Blur",
        "0": "Exit Program"
    }

    while True:
        print("\nChoose a blur effect (0 to exit):")
        for k, v in effects.items():
            print(f"{k}: {v}")

        choice = input("\nEnter number of effect: ").strip()

        if choice == "0":
            print("Exiting program...")
            break

        if choice not in effects:
            print("❌ Invalid choice. Try again.")
            continue

        if choice == "1":
            result = linear_blur(img, ksize=(71, 71), direction='vertical', strip_width=0.3)
        elif choice == "2":
            result = linear_blur(img, ksize=(71, 71), direction='horizontal', strip_width=0.3)
        elif choice == "3":
            result = linear_blur(img, ksize=(71, 71), direction='angle', angle=45, strip_width=0.25)
        elif choice == "4":
            result = radial_blur(img, ksize=(91, 91), inner_radius_ratio=0.2, outer_radius_ratio=0.7)
        elif choice == "5":
            result = circular_blur(img, ksize=(71, 71), radius_ratio=0.3, feather=61)
        elif choice == "6":
            result = oval_blur(img, ksize=(71, 71), width_ratio=0.5, height_ratio=0.3, feather=61)
        elif choice == "7":
            result = focus_blur(img, ksize=(71, 71), focus_region='center')

        save = input("\n💾 Save result? (y/n): ")
        if save.lower() == 'y':
            filename = f"output_{effects.get(choice).replace(' ', '_').lower()}.jpg"
            cv2.imwrite(filename, result)
            print(f"✅ Saved as {filename}")

