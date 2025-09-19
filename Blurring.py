import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import cv2
import math
import os
import platform

def open_image(path):
    """Open image in default viewer"""
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":  # macOS
        os.system(f"open '{path}'")
    else:  # Linux
        os.system(f"xdg-open '{path}'")

def load_image(path):
    """Load image and convert to numpy array"""
    try:
        img = Image.open(path)
        return np.array(img)
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None

def save_image(arr, filename):
    """Save numpy array as image"""
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(filename)
    print(f"✅ Saved: {filename}")

# ==============================================
# 1. OVAL BLUR - Blur in elliptical pattern
# ==============================================
def oval_blur(image_arr, center_x=None, center_y=None, radius_x=200, radius_y=100, max_blur=15, falloff=2.0):
    """
    Apply oval/elliptical blur effect
    
    Args:
        image_arr: Input image as numpy array
        center_x, center_y: Center of oval (None for image center)
        radius_x, radius_y: Oval radii in pixels
        max_blur: Maximum blur strength
        falloff: How quickly blur falls off (higher = sharper edge)
    """
    h, w = image_arr.shape[:2]
    
    if center_x is None:
        center_x = w // 2
    if center_y is None:
        center_y = h // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    
    # Calculate elliptical distance
    dx = (x - center_x) / radius_x
    dy = (y - center_y) / radius_y
    distance = np.sqrt(dx**2 + dy**2)
    
    # Create blur mask (0 = no blur, 1 = max blur)
    blur_mask = np.clip(distance - 1.0, 0, None) ** falloff
    blur_mask = np.clip(blur_mask, 0, 1)
    
    # Apply different blur levels
    result = image_arr.copy()
    
    for blur_level in range(1, max_blur + 1):
        # Create blur kernel
        kernel_size = blur_level * 2 + 1
        kernel = cv2.getGaussianKernel(kernel_size, blur_level / 3.0)
        kernel = np.outer(kernel, kernel)
        
        # Apply blur
        if len(image_arr.shape) == 3:
            blurred = cv2.filter2D(image_arr, -1, kernel)
        else:
            blurred = cv2.filter2D(image_arr, -1, kernel)
        
        # Blend based on mask
        mask_level = (blur_mask * max_blur >= blur_level).astype(float)
        mask_level = cv2.GaussianBlur(mask_level.astype(np.float32), (5, 5), 1)
        
        if len(image_arr.shape) == 3:
            for c in range(image_arr.shape[2]):
                result[:, :, c] = result[:, :, c] * (1 - mask_level) + blurred[:, :, c] * mask_level
        else:
            result = result * (1 - mask_level) + blurred * mask_level
    
    return result.astype(np.uint8)

# ==============================================
# 2. RADIAL BLUR - Blur radiating from center
# ==============================================
def radial_blur(image_arr, center_x=None, center_y=None, strength=10, samples=8):
    """
    Apply radial blur effect (zoom blur)
    
    Args:
        image_arr: Input image as numpy array
        center_x, center_y: Center point for radial blur
        strength: Blur strength (distance in pixels)
        samples: Number of samples for averaging
    """
    h, w = image_arr.shape[:2]
    
    if center_x is None:
        center_x = w // 2
    if center_y is None:
        center_y = h // 2
    
    result = np.zeros_like(image_arr, dtype=np.float32)
    
    # Sample points along radial lines
    for i in range(samples):
        # Calculate offset from center
        factor = (i - samples // 2) * strength / samples
        
        # Create coordinate maps
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Calculate direction from center
        dx = x_coords - center_x
        dy = y_coords - center_y
        
        # Normalize direction vectors
        distance = np.sqrt(dx**2 + dy**2)
        distance[distance == 0] = 1  # Avoid division by zero
        
        # Calculate sample positions
        sample_x = x_coords + (dx / distance) * factor
        sample_y = y_coords + (dy / distance) * factor
        
        # Clamp coordinates
        sample_x = np.clip(sample_x, 0, w - 1)
        sample_y = np.clip(sample_y, 0, h - 1)
        
        # Bilinear interpolation
        x0 = sample_x.astype(int)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = sample_y.astype(int)
        y1 = np.clip(y0 + 1, 0, h - 1)
        
        wx = sample_x - x0
        wy = sample_y - y0
        
        if len(image_arr.shape) == 3:
            for c in range(image_arr.shape[2]):
                # Bilinear interpolation for each channel
                interpolated = (image_arr[y0, x0, c] * (1 - wx) * (1 - wy) +
                              image_arr[y0, x1, c] * wx * (1 - wy) +
                              image_arr[y1, x0, c] * (1 - wx) * wy +
                              image_arr[y1, x1, c] * wx * wy)
                result[:, :, c] += interpolated
        else:
            interpolated = (image_arr[y0, x0] * (1 - wx) * (1 - wy) +
                          image_arr[y0, x1] * wx * (1 - wy) +
                          image_arr[y1, x0] * (1 - wx) * wy +
                          image_arr[y1, x1] * wx * wy)
            result += interpolated
    
    result /= samples
    return result.astype(np.uint8)

# ==============================================
# 3. LINEAR BLUR - Motion blur in a direction
# ==============================================
def linear_blur(image_arr, angle=0, length=20, samples=15):
    """
    Apply linear/motion blur effect
    
    Args:
        image_arr: Input image as numpy array
        angle: Blur direction in degrees
        length: Blur length in pixels
        samples: Number of samples for averaging
    """
    h, w = image_arr.shape[:2]
    
    # Convert angle to radians
    angle_rad = math.radians(angle)
    
    # Calculate direction vector
    dx = math.cos(angle_rad) * length
    dy = math.sin(angle_rad) * length
    
    result = np.zeros_like(image_arr, dtype=np.float32)
    
    # Sample along the blur line
    for i in range(samples):
        # Calculate offset
        t = (i - samples // 2) / samples
        offset_x = dx * t
        offset_y = dy * t
        
        # Create shifted coordinates
        y_coords, x_coords = np.ogrid[:h, :w]
        sample_x = x_coords + offset_x
        sample_y = y_coords + offset_y
        
        # Create valid mask
        valid_mask = ((sample_x >= 0) & (sample_x < w - 1) & 
                     (sample_y >= 0) & (sample_y < h - 1))
        
        # Bilinear interpolation
        x0 = np.clip(sample_x.astype(int), 0, w - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = np.clip(sample_y.astype(int), 0, h - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        
        wx = np.clip(sample_x - x0, 0, 1)
        wy = np.clip(sample_y - y0, 0, 1)
        
        if len(image_arr.shape) == 3:
            for c in range(image_arr.shape[2]):
                interpolated = (image_arr[y0, x0, c] * (1 - wx) * (1 - wy) +
                              image_arr[y0, x1, c] * wx * (1 - wy) +
                              image_arr[y1, x0, c] * (1 - wx) * wy +
                              image_arr[y1, x1, c] * wx * wy)
                result[:, :, c] += interpolated * valid_mask
        else:
            interpolated = (image_arr[y0, x0] * (1 - wx) * (1 - wy) +
                          image_arr[y0, x1] * wx * (1 - wy) +
                          image_arr[y1, x0] * (1 - wx) * wy +
                          image_arr[y1, x1] * wx * wy)
            result += interpolated * valid_mask
    
    # Normalize by number of valid samples
    sample_count = np.zeros((h, w))
    for i in range(samples):
        t = (i - samples // 2) / samples
        offset_x = dx * t
        offset_y = dy * t
        y_coords, x_coords = np.ogrid[:h, :w]
        sample_x = x_coords + offset_x
        sample_y = y_coords + offset_y
        valid_mask = ((sample_x >= 0) & (sample_x < w - 1) & 
                     (sample_y >= 0) & (sample_y < h - 1))
        sample_count += valid_mask
    
    sample_count[sample_count == 0] = 1
    
    if len(image_arr.shape) == 3:
        for c in range(image_arr.shape[2]):
            result[:, :, c] /= sample_count
    else:
        result /= sample_count
    
    return result.astype(np.uint8)

# ==============================================
# 4. CIRCULAR BLUR - Blur in circular pattern
# ==============================================
def circular_blur(image_arr, center_x=None, center_y=None, radius=100, max_blur=15):
    """
    Apply circular blur effect (blur increases with distance from center)
    
    Args:
        image_arr: Input image as numpy array
        center_x, center_y: Center of circle
        radius: Radius where blur starts
        max_blur: Maximum blur strength
    """
    h, w = image_arr.shape[:2]
    
    if center_x is None:
        center_x = w // 2
    if center_y is None:
        center_y = h // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    
    # Calculate distance from center
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create blur mask
    blur_strength = np.clip((distance - radius) / radius, 0, 1) * max_blur
    
    result = image_arr.copy().astype(np.float32)
    
    # Apply blur in layers
    for y_pos in range(h):
        for x_pos in range(w):
            blur_val = int(blur_strength[y_pos, x_pos])
            
            if blur_val > 0:
                # Define region around current pixel
                kernel_size = blur_val * 2 + 1
                half_kernel = blur_val
                
                y_start = max(0, y_pos - half_kernel)
                y_end = min(h, y_pos + half_kernel + 1)
                x_start = max(0, x_pos - half_kernel)
                x_end = min(w, x_pos + half_kernel + 1)
                
                # Extract region
                region = image_arr[y_start:y_end, x_start:x_end]
                
                if region.size > 0:
                    if len(image_arr.shape) == 3:
                        blurred_pixel = np.mean(region, axis=(0, 1))
                        result[y_pos, x_pos] = blurred_pixel
                    else:
                        result[y_pos, x_pos] = np.mean(region)
    
    return result.astype(np.uint8)

# ==============================================
# OPTIMIZED CIRCULAR BLUR (using convolution)
# ==============================================
def circular_blur_optimized(image_arr, center_x=None, center_y=None, radius=100, max_blur=15, steps=5):
    """
    Optimized circular blur using Gaussian blur with varying strengths
    """
    h, w = image_arr.shape[:2]
    
    if center_x is None:
        center_x = w // 2
    if center_y is None:
        center_y = h // 2
    
    # Create distance map
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Normalize distance
    normalized_distance = np.clip((distance - radius) / radius, 0, 1)
    
    result = image_arr.copy().astype(np.float32)
    
    # Apply blur in steps
    for step in range(steps):
        blur_level = int((step + 1) * max_blur / steps)
        if blur_level < 1:
            continue
            
        # Create Gaussian kernel
        kernel_size = blur_level * 2 + 1
        blurred = cv2.GaussianBlur(image_arr, (kernel_size, kernel_size), blur_level / 3.0)
        
        # Create mask for this blur level
        step_threshold = step / steps
        next_threshold = (step + 1) / steps
        
        mask = ((normalized_distance >= step_threshold) & 
                (normalized_distance < next_threshold)).astype(np.float32)
        
        # Smooth the mask edges
        mask = cv2.GaussianBlur(mask, (5, 5), 1)
        
        # Blend
        if len(image_arr.shape) == 3:
            for c in range(image_arr.shape[2]):
                result[:, :, c] = (result[:, :, c] * (1 - mask) + 
                                 blurred[:, :, c] * mask)
        else:
            result = result * (1 - mask) + blurred * mask
    
    return result.astype(np.uint8)

# ==============================================
# DEMO FUNCTION
# ==============================================
def demo_all_blurs(input_path="input.jpg"):
    """Demonstrate all blur effects"""
    print("🎨 Advanced Blur Effects Demo")
    print("=" * 40)
    
    # Load image
    image_arr = load_image(input_path)
    if image_arr is None:
        return
    
    h, w = image_arr.shape[:2]
    print(f"📷 Image loaded: {w}x{h}")
    
    # 1. Oval Blur
    print("🔄 Applying oval blur...")
    oval_result = oval_blur(image_arr, radius_x=w//4, radius_y=h//6, max_blur=12)
    save_image(oval_result, "oval_blur.jpg")
    
    # 2. Radial Blur
    print("🔄 Applying radial blur...")
    radial_result = radial_blur(image_arr, strength=15, samples=12)
    save_image(radial_result, "radial_blur.jpg")
    
    # 3. Linear Blur (horizontal)
    print("🔄 Applying linear blur (horizontal)...")
    linear_h_result = linear_blur(image_arr, angle=0, length=25)
    save_image(linear_h_result, "linear_blur_horizontal.jpg")
    
    # 4. Linear Blur (diagonal)
    print("🔄 Applying linear blur (diagonal)...")
    linear_d_result = linear_blur(image_arr, angle=45, length=20)
    save_image(linear_d_result, "linear_blur_diagonal.jpg")
    
    # 5. Circular Blur
    print("🔄 Applying circular blur...")
    circular_result = circular_blur_optimized(image_arr, radius=min(w,h)//4, max_blur=12)
    save_image(circular_result, "circular_blur.jpg")
    
    print("\n✅ All blur effects generated!")
    print("📁 Output files:")
    print("   • oval_blur.jpg")
    print("   • radial_blur.jpg")
    print("   • linear_blur_horizontal.jpg")
    print("   • linear_blur_diagonal.jpg")
    print("   • circular_blur.jpg")

# ==============================================
# PREVIEW AND SAVE FUNCTION
# ==============================================
def preview_and_save(result_image, effect_name):
    """Preview effect and ask user if they want to save"""
    # Save preview
    preview_filename = f"preview_{effect_name}.jpg"
    save_image(result_image, preview_filename)
    print(f"👁️  Opening preview: {preview_filename}")
    open_image(preview_filename)
    
    # Ask user decision
    while True:
        decision = input("\n🤔 Do you like this effect? (y)es to save / (n)o to discard / (r)etry with different settings: ").strip().lower()
        
        if decision in ['y', 'yes']:
            final_filename = f"final_{effect_name}.jpg"
            save_image(result_image, final_filename)
            print(f"✅ Final image saved as: {final_filename}")
            open_image(final_filename)
            return 'saved'
            
        elif decision in ['n', 'no']:
            print("❌ Effect discarded.")
            # Clean up preview file
            try:
                os.remove(preview_filename)
            except:
                pass
            return 'discarded'
            
        elif decision in ['r', 'retry']:
            print("🔄 Let's try different settings...")
            # Clean up preview file
            try:
                os.remove(preview_filename)
            except:
                pass
            return 'retry'
            
        else:
            print("❌ Please enter 'y' (yes), 'n' (no), or 'r' (retry)")

# ==============================================
# INTERACTIVE FUNCTION
# ==============================================
def interactive_blur():
    """Interactive blur effect selector with preview functionality"""
    print("🎨 Interactive Blur Effects with Preview")
    print("=" * 50)
    print("📝 Workflow: Choose effect → Set parameters → Preview → Save/Discard/Retry")
    print("=" * 50)
    
    input_path = input("📂 Enter image path (or press Enter for 'input.jpg'): ").strip()
    if not input_path:
        input_path = "input.jpg"
    
    image_arr = load_image(input_path)
    if image_arr is None:
        return
    
    h, w = image_arr.shape[:2]
    
    while True:
        print(f"\n🖼️  Working with image: {w}x{h}")
        print("Select blur type:")
        print("1. Oval Blur")
        print("2. Radial Blur") 
        print("3. Linear Blur")
        print("4. Circular Blur")
        print("5. Demo All Effects")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-5): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
            
        elif choice == "1":
            while True:
                print("\n🔵 Oval Blur Settings:")
                rx = int(input(f"Radius X (default {w//4}): ") or w//4)
                ry = int(input(f"Radius Y (default {h//6}): ") or h//6)
                blur_strength = int(input("Blur strength 1-20 (default 10): ") or 10)
                falloff = float(input("Edge falloff 0.5-5.0 (default 2.0): ") or 2.0)
                
                print("🔄 Generating oval blur effect...")
                result = oval_blur(image_arr, radius_x=rx, radius_y=ry, max_blur=blur_strength, falloff=falloff)
                
                decision = preview_and_save(result, "oval_blur")
                if decision != 'retry':
                    break
            
        elif choice == "2":
            while True:
                print("\n🌀 Radial Blur Settings:")
                strength = int(input("Blur strength 1-30 (default 15): ") or 15)
                samples = int(input("Quality samples 4-20 (default 10): ") or 10)
                
                print("🔄 Generating radial blur effect...")
                result = radial_blur(image_arr, strength=strength, samples=samples)
                
                decision = preview_and_save(result, "radial_blur")
                if decision != 'retry':
                    break
            
        elif choice == "3":
            while True:
                print("\n➡️ Linear Blur Settings:")
                angle = int(input("Angle in degrees (default 0): ") or 0)
                length = int(input("Blur length 1-50 (default 20): ") or 20)
                samples = int(input("Quality samples 5-25 (default 15): ") or 15)
                
                print("🔄 Generating linear blur effect...")
                result = linear_blur(image_arr, angle=angle, length=length, samples=samples)
                
                decision = preview_and_save(result, "linear_blur")
                if decision != 'retry':
                    break
            
        elif choice == "4":
            while True:
                print("\n⭕ Circular Blur Settings:")
                radius = int(input(f"Inner radius (default {min(w,h)//4}): ") or min(w,h)//4)
                max_blur = int(input("Max blur strength 1-20 (default 12): ") or 12)
                steps = int(input("Blur quality steps 3-10 (default 5): ") or 5)
                
                print("🔄 Generating circular blur effect...")
                result = circular_blur_optimized(image_arr, radius=radius, max_blur=max_blur, steps=steps)
                
                decision = preview_and_save(result, "circular_blur")
                if decision != 'retry':
                    break
            
        elif choice == "5":
            print("\n🎬 Generating demo of all effects...")
            demo_all_blurs(input_path)
            
            print("\n👁️  Opening all demo files...")
            demo_files = ["oval_blur.jpg", "radial_blur.jpg", "linear_blur_horizontal.jpg", 
                         "linear_blur_diagonal.jpg", "circular_blur.jpg"]
            
            for filename in demo_files:
                if os.path.exists(filename):
                    open_image(filename)
            
            print("\n💾 Demo files generated. You can manually save the ones you like!")
        
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    # You can run either:
    # 1. Demo all effects: demo_all_blurs()
    # 2. Interactive mode: interactive_blur()
    
    interactive_blur()