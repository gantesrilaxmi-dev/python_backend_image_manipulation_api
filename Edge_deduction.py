import cv2
import numpy as np

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
# Menu Options
# -------------------------
while True:
    print("\nChoose an operation:")
    print("1. Edge Detection")
    print("2. Foreground Extraction")
    print("3. Background Removal")
    print("4. Overlay Edges on Foreground")
    print("5. Exit")
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
        break
    else:
        print("Invalid option, try again.")

cv2.waitKey(0)
cv2.destroyAllWindows()




