import cv2
import numpy as np
from sklearn.cluster import KMeans

# -------------------------
# Load Image
# -------------------------
image_path = "input.jpg"  # Replace with your image path
img = cv2.imread(image_path)
original = img.copy()
mask = np.zeros(img.shape[:2], np.uint8)

# -------------------------
# Edge Detection Function
# -------------------------
def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    cv2.imshow("Edges", edges)
    cv2.imwrite("edges.jpg", edges)
    print("Edges saved as 'edges.jpg'")
    return edges, gray

# -------------------------
# Foreground Extraction Function
# -------------------------
def grabcut_foreground(image):
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    height, width = image.shape[:2]
    rect = (10, 10, width - 20, height - 20)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    foreground = image * mask2[:, :, np.newaxis]
    cv2.imshow("Foreground", foreground)
    cv2.imwrite("foreground_grabcut.jpg", foreground)
    print("Foreground saved as 'foreground_grabcut.jpg'")
    return foreground, mask2

# -------------------------
# Background Removal Function
# -------------------------
def background_removal(foreground, mask2):
    b, g, r = cv2.split(foreground)
    alpha = mask2 * 255
    background_removed = cv2.merge([b, g, r, alpha])
    cv2.imshow("Background Removed", background_removed)
    cv2.imwrite("background_removed_grabcut.png", background_removed)
    print("Background removed image saved as 'background_removed_grabcut.png'")
    return background_removed

# -------------------------
# Overlay Edges Function
# -------------------------
def overlay_edges(foreground, edges):
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(foreground, 0.8, edges_colored, 0.2, 0)
    cv2.imshow("Foreground with Edges", overlay)
    cv2.imwrite("foreground_edges_overlay.jpg", overlay)
    print("Overlay saved as 'foreground_edges_overlay.jpg'")

# -------------------------
# Detect Dominant Colors Function
# -------------------------
def detect_dominant_colors(image, k=5):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_flat = img_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42).fit(img_flat)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

# -------------------------
# Apply Selected Color as Background
# -------------------------
def apply_background_color(image, mask2, bg_color=(0, 255, 0)):
    background = np.full(image.shape, bg_color[::-1], dtype=np.uint8)
    foreground = image * mask2[:, :, np.newaxis]
    combined = foreground + (background * (1 - mask2[:, :, np.newaxis]))
    cv2.imshow("Background Replaced", combined)
    cv2.imwrite("background_with_color.jpg", combined)
    print(f"Background replaced with color {bg_color}")
    return combined

# -------------------------
# Add Adaptive Shadow Behind Foreground
# -------------------------
def add_shadow(image, mask2, background_image=None, offset=(15, 15), blur_size=25):
    shadow = np.zeros_like(image)

    # Shift mask for shadow
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    shifted_mask = cv2.warpAffine(mask2, M, (mask2.shape[1], mask2.shape[0]))

    # Use provided background or white by default
    if background_image is None:
        background = np.full(image.shape, (255, 255, 255), dtype=np.uint8)
    else:
        if background_image.shape[:2] != image.shape[:2]:
            background = cv2.resize(background_image, (image.shape[1], image.shape[0]))
        else:
            background = background_image.copy()

    # Determine shadow color based on background brightness
    avg_brightness = np.mean(background[shifted_mask == 1]) if np.any(shifted_mask) else 127
    shadow_intensity = 50 if avg_brightness > 127 else 200
    shadow_color = (shadow_intensity, shadow_intensity, shadow_intensity)

    shadow[shifted_mask == 1] = shadow_color
    shadow = cv2.GaussianBlur(shadow, (blur_size, blur_size), 0)

    # Blend shadow with background
    shadow_alpha = 0.5
    with_shadow = cv2.addWeighted(background, 1, shadow, shadow_alpha, 0)

    # Overlay foreground
    foreground = image * mask2[:, :, np.newaxis]
    combined = foreground + (with_shadow * (1 - mask2[:, :, np.newaxis]))

    cv2.imshow("Foreground with Shadow", combined)
    cv2.imwrite("foreground_with_shadow.jpg", combined)
    print("Foreground with shadow applied on current background")
    return combined

# -------------------------
# Menu Options
# -------------------------
while True:
    print("\nChoose an operation:")
    print("1. Edge Detection")
    print("2. Foreground Extraction")
    print("3. Background Removal")
    print("4. Overlay Edges on Foreground")
    print("5. Color Deduction & Background Replace")
    print("6. Add Shadow Behind Foreground")
    print("7. Exit")
    choice = input("Enter option number: ")

    if choice == "1":
        edges, gray = edge_detection(img)
    elif choice == "2":
        foreground, mask2 = grabcut_foreground(img)
    elif choice == "3":
        try:
            background_removed = background_removal(foreground, mask2)
        except NameError:
            print("Run Foreground Extraction first!")
    elif choice == "4":
        try:
            overlay_edges(foreground, edges)
        except NameError:
            print("Run Edge Detection and Foreground Extraction first!")
    elif choice == "5":
        try:
            colors = detect_dominant_colors(img, k=5)
            print("Detected Colors (R,G,B):")
            for i, c in enumerate(colors):
                print(f"{i+1}. {tuple(c)}")
            idx = int(input("Select color number to use as background: ")) - 1
            selected_color = tuple(colors[idx])
            output = apply_background_color(img, mask2, bg_color=selected_color)
        except NameError:
            print("Run Foreground Extraction first!")
    elif choice == "6":
        try:
            if 'output' in globals():
                shadowed = add_shadow(img, mask2, background_image=output)
            else:
                shadowed = add_shadow(img, mask2)
        except NameError:
            print("Run Foreground Extraction first!")
    elif choice == "7":
        break
    else:
        print("Invalid option, try again.")
