import os
import cv2
import numpy as np
import tensorflow as tf
import time
import math

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class PhotographyAssistant:
    def __init__(self):
        # Load TFLite pose model
        self.pose_model_path = r"C:\Users\gante\OneDrive\Desktop\cam_web\models\pose_estimation.tflite"
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.pose_model_path)
            self.interpreter.allocate_tensors()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.interpreter = None

        # Pose keypoints and edges
        self.KEYPOINT_DICT = {
            'nose':0,'left_eye':1,'right_eye':2,'left_ear':3,'right_ear':4,
            'left_shoulder':5,'right_shoulder':6,'left_elbow':7,'right_elbow':8,
            'left_wrist':9,'right_wrist':10,'left_hip':11,'right_hip':12,
            'left_knee':13,'right_knee':14,'left_ankle':15,'right_ankle':16
        }
        self.EDGES = {
            (0,1):(255,0,255),(0,2):(0,255,255),(1,3):(255,0,255),(2,4):(0,255,255),
            (0,5):(255,0,255),(0,6):(0,255,255),(5,7):(255,0,255),(7,9):(255,0,255),
            (6,8):(0,255,255),(8,10):(0,255,255),(5,6):(0,255,0),(5,11):(255,0,255),
            (6,12):(0,255,255),(11,12):(0,255,0),(11,13):(255,0,255),(13,15):(255,0,255),
            (12,14):(0,255,255),(14,16):(0,255,255)
        }

        # Camera & person parameters
        self.frame_size = (640,480)
        self.confidence_threshold = 0.2
        self.H_current = 5.5  # Default height in feet (average human height)
        self.focal_length = 1000
        
        # Distance thresholds
        self.MIN_DISTANCE = 1.5  # Minimum required distance in feet
        self.PORTRAIT_MAX_DISTANCE = 5.5  # Maximum distance for portrait mode in feet
        
        # Camera height estimation
        self.estimated_ground_height = None
        self.is_portrait_mode = False
        
        # UI Mode Management
        self.current_mode = "live_camera"  # Default mode - live camera view
        
        # Capture state
        self.captured_frame = None
        self.captured_outline = None
        self.last_keypoints = None
        self.last_measurements = None
        self.is_captured = False

    # ------------------------------
    # UI Mode Management
    # ------------------------------
    def switch_mode(self, direction="next"):
        """Switch between different UI modes"""
        if direction == "next":
            self.mode_index = (self.mode_index + 1) % len(self.view_modes)
        else:
            self.mode_index = (self.mode_index - 1) % len(self.view_modes)
        
        self.current_mode = self.view_modes[self.mode_index]
        print(f"Switched to mode: {self.current_mode}")

    def create_detecting_calculations_view(self, frame, detected_joints, pixel_height, 
                                         object_distance, photography_mode, orientation, ground_height):
        """Create the detecting and calculations part view"""
        calc_frame = frame.copy()
        
        # Create a semi-transparent overlay for calculations
        overlay = np.zeros_like(calc_frame)
        
        # Left side - Detection info
        y_start = 50
        line_height = 25
        
        # Title
        cv2.putText(overlay, "DETECTION & CALCULATIONS", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_start += 40
        
        # Joint detection status with detailed breakdown
        cv2.putText(overlay, "Joint Detection Status:", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_start += line_height
        
        if detected_joints:
            for joint, detected in detected_joints.items():
                status = "DETECTED" if detected else "NOT DETECTED"
                color = (0, 255, 0) if detected else (0, 0, 255)
                cv2.putText(overlay, f"  {joint.upper()}: {status}", (30, y_start), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_start += 18
        
        # Right side - Measurements
        right_x = 350
        y_right = 90
        
        cv2.putText(overlay, "MEASUREMENTS:", (right_x, y_right), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_right += line_height
        
        measurements = [
            f"Object Distance: {object_distance:.2f} ft" if object_distance else "Object Distance: N/A",
            f"Pixel Height: {pixel_height:.2f} px" if pixel_height else "Pixel Height: N/A",
            f"Ground Height: {ground_height:.2f} ft" if ground_height else "Ground Height: N/A",
            f"Mode: {photography_mode}",
            f"Orientation: {orientation}"
        ]
        
        for measurement in measurements:
            cv2.putText(overlay, measurement, (right_x, y_right), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_right += 20
        
        # Blend overlay with original frame
        alpha = 0.7
        cv2.addWeighted(calc_frame, 1 - alpha, overlay, alpha, 0, calc_frame)
        
        return calc_frame

    def create_outline_generation_view(self, outline_frame, keypoints):
        """Create outline generation after hitting capture button"""
        if outline_frame is None:
            return np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        
        # Add generation effect
        gen_frame = outline_frame.copy()
        
        # Add title
        cv2.putText(gen_frame, "OUTLINE GENERATION", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add processing indicator
        cv2.putText(gen_frame, "Processing pose outline...", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return gen_frame

    def get_mode_display_name(self):
        """Get display name for current mode"""
        mode_names = {
            "camera_on_apply": "Camera On Apply",
            "detecting_calculations": "Detecting & Calculations", 
            "outline_view": "Outline View Mode",
            "image_view": "Image View Mode"
        }
        return mode_names.get(self.current_mode, self.current_mode)

    # ------------------------------
    # Pose Utilities
    # ------------------------------
    def preprocess_frame(self, frame):
        img = cv2.resize(frame, (192,192))
        img = np.expand_dims(img, axis=0)
        return img.astype(np.uint8)

    def run_pose_inference(self, frame):
        if not self.interpreter: return None
        input_data = self.preprocess_frame(frame)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(output_details[0]['index'])

    def draw_pose(self, frame, keypoints):
        h, w = frame.shape[:2]
        
        # Draw keypoints with different colors for different body parts
        joint_colors = {
            # Face joints - Red
            0: (0, 0, 255), 1: (0, 0, 255), 2: (0, 0, 255), 3: (0, 0, 255), 4: (0, 0, 255),
            # Shoulders - Green
            5: (0, 255, 0), 6: (0, 255, 0),
            # Elbows - Blue
            7: (255, 0, 0), 8: (255, 0, 0),
            # Wrists/Palms - Yellow
            9: (0, 255, 255), 10: (0, 255, 255),
            # Hips - Magenta
            11: (255, 0, 255), 12: (255, 0, 255),
            # Knees - Cyan
            13: (255, 255, 0), 14: (255, 255, 0),
            # Feet/Ankles - White
            15: (255, 255, 255), 16: (255, 255, 255)
        }
        
        for i, k in enumerate(keypoints):
            y, x, score = k
            if score > self.confidence_threshold:
                color = joint_colors.get(i, (0, 255, 0))
                cv2.circle(frame, (int(x*w), int(y*h)), 4, color, -1)
        
        # Draw skeleton connections
        for edge, color in self.EDGES.items():
            pt1, pt2 = edge
            y1, x1, s1 = keypoints[pt1]
            y2, x2, s2 = keypoints[pt2]
            if s1 > self.confidence_threshold and s2 > self.confidence_threshold:
                cv2.line(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), color, 2)
        
        return frame

    def detect_all_joints(self, keypoints):
        """Detect all required joints and return their status"""
        required_joints = {
            'face': [0],  # nose (representing face)
            'left_shoulder': [5],
            'right_shoulder': [6],
            'left_elbow': [7],
            'right_elbow': [8],
            'left_palm': [9],  # left wrist (representing palm)
            'right_palm': [10],  # right wrist (representing palm)
            'left_hip': [11],
            'right_hip': [12],
            'left_knee': [13],
            'right_knee': [14],
            'left_foot': [15],  # left ankle (representing foot)
            'right_foot': [16]   # right ankle (representing foot)
        }
        
        detected_joints = {}
        for joint_name, indices in required_joints.items():
            detected = False
            for idx in indices:
                if keypoints[idx, 2] > self.confidence_threshold:
                    detected = True
                    break
            detected_joints[joint_name] = detected
        
        return detected_joints

    def calculate_height_distance(self, keypoints):
        """Calculate both object distance and estimate camera height from ground"""
        head_indices = [0, 1, 2]  # nose, eyes
        foot_indices = [15, 16]   # ankles
        hip_indices = [11, 12]    # hips
        shoulder_indices = [5, 6]  # shoulders
        
        h_vals, f_vals, hip_vals, shoulder_vals = [], [], [], []
        
        # Get head points
        for i in head_indices:
            if keypoints[i, 2] > self.confidence_threshold:
                h_vals.append(keypoints[i, 0])
        
        # Get foot points
        for i in foot_indices:
            if keypoints[i, 2] > self.confidence_threshold:
                f_vals.append(keypoints[i, 0])
        
        # Get hip points for height estimation
        for i in hip_indices:
            if keypoints[i, 2] > self.confidence_threshold:
                hip_vals.append(keypoints[i, 0])
                
        # Get shoulder points
        for i in shoulder_indices:
            if keypoints[i, 2] > self.confidence_threshold:
                shoulder_vals.append(keypoints[i, 0])
        
        if not h_vals:
            return None, None, None
        
        sitting = False
        if not f_vals:
            # Person might be sitting - estimate foot position
            f_vals = [h_vals[0] + 0.6]
            sitting = True
        
        # Calculate pixel height of person
        pixel_height = abs(np.mean(f_vals) - np.mean(h_vals)) * self.frame_size[1]
        
        # Calculate object distance (camera to person)
        person_height_multiplier = 0.6 if sitting else 1.0
        object_distance = (self.H_current * person_height_multiplier * self.focal_length) / pixel_height
        
        # Enhanced ground height estimation
        ground_height = None
        if f_vals and object_distance:
            # Get foot position in pixels from top of frame
            foot_pixel_pos = np.mean(f_vals) * self.frame_size[1]
            frame_center_y = self.frame_size[1] / 2
            
            # Calculate vertical angle from camera center to foot level
            vertical_angle = math.atan((foot_pixel_pos - frame_center_y) / self.focal_length)
            
            if not sitting:
                # For standing person, calculate camera height above ground
                # If feet appear below frame center, camera is looking down (elevated)
                # If feet appear above frame center, camera is looking up (low position)
                
                # Distance to ground plane at person's location
                ground_distance = object_distance
                
                # Height difference calculation
                height_difference = ground_distance * math.tan(vertical_angle)
                
                # Assume average person height is 5.5 feet and camera is at chest level when holding
                base_camera_height = 4.5  # feet (typical handheld camera height)
                
                # Calculate actual ground height
                ground_height = base_camera_height + height_difference
                
                # Clamp to reasonable values
                ground_height = max(0.5, min(12.0, ground_height))
                
                # If we have shoulder data, refine the calculation
                if shoulder_vals and hip_vals:
                    shoulder_pixel_pos = np.mean(shoulder_vals) * self.frame_size[1]
                    hip_pixel_pos = np.mean(hip_vals) * self.frame_size[1]
                    
                    # Typical human proportions: shoulder to foot is about 75% of total height
                    shoulder_to_foot_pixels = foot_pixel_pos - shoulder_pixel_pos
                    estimated_person_height_pixels = shoulder_to_foot_pixels / 0.75
                    
                    # Refine distance calculation using shoulder reference
                    refined_distance = (self.H_current * self.focal_length) / estimated_person_height_pixels
                    
                    if abs(refined_distance - object_distance) < object_distance * 0.3:  # Within 30% tolerance
                        # Use refined calculation
                        height_difference_refined = refined_distance * math.tan(vertical_angle)
                        ground_height = base_camera_height + height_difference_refined
                        ground_height = max(0.5, min(12.0, ground_height))
            else:
                # For sitting person, estimate ground height differently
                # Assume sitting person's hip is about 1.5 feet from ground
                sitting_height = 1.5
                if hip_vals:
                    hip_pixel_pos = np.mean(hip_vals) * self.frame_size[1]
                    hip_vertical_angle = math.atan((hip_pixel_pos - frame_center_y) / self.focal_length)
                    hip_height_difference = object_distance * math.tan(hip_vertical_angle)
                    ground_height = 4.5 + hip_height_difference - sitting_height  # Subtract sitting height
                else:
                    ground_height = 3.0  # Default for sitting estimation
                
                ground_height = max(0.5, min(8.0, ground_height))
        
        return pixel_height, object_distance, ground_height

    def classify_photography_mode(self, distance):
        """Classify as portrait or normal photography based on distance"""
        if distance is None:
            return "Unknown"
        
        if distance < self.MIN_DISTANCE:
            return "Too Close"
        elif self.MIN_DISTANCE <= distance <= self.PORTRAIT_MAX_DISTANCE:
            return "Portrait"
        else:
            return "Normal"

    def detect_orientation(self, frame):
        """Detect if frame is in landscape or portrait orientation"""
        h, w = frame.shape[:2]
        if h > w:
            return "Portrait (Vertical)"
        else:
            return "Landscape (Horizontal)"

    # ------------------------------
    # Camera mode + capture/navigation
    # ------------------------------
    def run_camera_mode(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])

        print("Photography Assistant Started!")
        print("Controls: 'c' capture | 'o' outline view | 'i' image view | 'a' live camera | 'q' quit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Initialize variables
            pixel_height, object_distance, ground_height = None, None, None
            detected_joints = {}
            photography_mode = "Unknown"
            orientation = self.detect_orientation(frame)
            
            # Create outline frame (black background)
            outline_frame = np.zeros_like(frame)

            # Run pose estimation
            pose_results = self.run_pose_inference(frame)
            if pose_results is not None:
                keypoints = pose_results[0][0]  # [17,3]
                
                # Draw pose on both views
                frame_with_pose = self.draw_pose(frame.copy(), keypoints)
                outline_frame = self.draw_pose(outline_frame, keypoints)
                
                # Detect all required joints
                detected_joints = self.detect_all_joints(keypoints)
                
                # Calculate distances and heights
                pixel_height, object_distance, ground_height = self.calculate_height_distance(keypoints)
                
                # Update estimated ground height
                if ground_height is not None:
                    self.estimated_ground_height = ground_height
                
                # Classify photography mode
                photography_mode = self.classify_photography_mode(object_distance)
                
                # Store current data
                self.last_keypoints = keypoints
                self.last_measurements = {
                    'pixel_height': pixel_height,
                    'object_distance': object_distance,
                    'ground_height': ground_height,
                    'photography_mode': photography_mode,
                    'orientation': orientation,
                    'detected_joints': detected_joints
                }

            # Display based on current mode
            if self.current_mode == "live_camera":
                # Live camera view with pose detection
                if pose_results is not None:
                    display_frame = frame_with_pose
                else:
                    display_frame = frame.copy()
                
                # Add live camera info overlay
                self.draw_live_camera_info(display_frame, detected_joints, pixel_height, 
                                         object_distance, photography_mode, orientation, ground_height)
                
            elif self.current_mode == "outline_view":
                # Show captured outline or live outline
                if self.is_captured and self.captured_outline is not None:
                    display_frame = self.captured_outline.copy()
                    cv2.putText(display_frame, "CAPTURED OUTLINE VIEW", (20, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    display_frame = outline_frame.copy()
                    cv2.putText(display_frame, "LIVE OUTLINE VIEW", (20, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Add measurements to outline view
                if object_distance:
                    cv2.putText(display_frame, f"Distance: {object_distance:.2f} ft", 
                               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if ground_height:
                    cv2.putText(display_frame, f"Ground Height: {ground_height:.2f} ft", 
                               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
            elif self.current_mode == "image_view":
                # Show captured image or live image
                if self.is_captured and self.captured_frame is not None:
                    display_frame = self.captured_frame.copy()
                    cv2.putText(display_frame, "CAPTURED IMAGE VIEW", (20, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    if pose_results is not None:
                        display_frame = frame_with_pose
                    else:
                        display_frame = frame.copy()
                    cv2.putText(display_frame, "LIVE IMAGE VIEW", (20, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Add detailed info to image view
                self.draw_detailed_info(display_frame, detected_joints, pixel_height, 
                                      object_distance, photography_mode, orientation, ground_height)

            # Add mode and control indicators
            mode_text = f"Mode: {self.current_mode.upper().replace('_', ' ')}"
            cv2.putText(display_frame, mode_text, 
                       (10, display_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            controls_text = "Controls: 'c' capture | 'o' outline | 'i' image | 'a' live | 'q' quit"
            cv2.putText(display_frame, controls_text, 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Show the display
            window_title = f"Photography Assistant - {mode_text}"
            cv2.imshow(window_title, display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Capture current frame
                if object_distance and object_distance >= self.MIN_DISTANCE:
                    if pose_results is not None:
                        self.captured_frame = frame_with_pose.copy()
                        self.captured_outline = outline_frame.copy()
                        self.is_captured = True
                        print("Frame captured successfully!")
                    else:
                        print("No person detected - capturing regular frame")
                        self.captured_frame = frame.copy()
                        self.captured_outline = np.zeros_like(frame)
                        self.is_captured = True
                else:
                    print("Cannot capture: Person not detected or too close (minimum 1.5 feet required)")
            elif key == ord('o'):
                # Switch to outline view
                self.current_mode = "outline_view"
                print("Switched to Outline View")
            elif key == ord('i'):
                # Switch to image view
                self.current_mode = "image_view"
                print("Switched to Image View")
            elif key == ord('a'):
                # Switch to live camera view
                self.current_mode = "live_camera"
                print("Switched to Live Camera")

        cap.release()
        cv2.destroyAllWindows()

    def draw_live_camera_info(self, frame, detected_joints, pixel_height, 
                            object_distance, photography_mode, orientation, ground_height):
        """Draw information overlay for live camera mode"""
        y_offset = 60
        line_height = 25
        
        # Basic measurements
        if object_distance:
            cv2.putText(frame, f"Distance: {object_distance:.2f} ft", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += line_height
            
            mode_color = (0, 255, 0) if photography_mode == "Portrait" else (0, 0, 255) if photography_mode == "Too Close" else (255, 0, 0)
            cv2.putText(frame, f"Mode: {photography_mode}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
            y_offset += line_height
        
        # Ground height
        if ground_height:
            cv2.putText(frame, f"Ground Height: {ground_height:.2f} ft", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y_offset += line_height
        
        # Joint count
        if detected_joints:
            detected_count = sum(detected_joints.values())
            total_joints = len(detected_joints)
            cv2.putText(frame, f"Joints: {detected_count}/{total_joints}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def draw_detailed_info(self, frame, detected_joints, pixel_height, 
                         object_distance, photography_mode, orientation, ground_height):
        """Draw detailed information overlay for image view"""
        # Create semi-transparent overlay
        overlay = np.zeros_like(frame)
        
        y_start = 60
        line_height = 20
        
        # Measurements section
        cv2.putText(overlay, "MEASUREMENTS:", (10, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_start += line_height + 5
        
        measurements = [
            f"Object Distance: {object_distance:.2f} ft" if object_distance else "Distance: N/A",
            f"Ground Height: {ground_height:.2f} ft" if ground_height else "Ground Height: N/A",
            f"Pixel Height: {pixel_height:.2f} px" if pixel_height else "Pixel Height: N/A",
            f"Photography Mode: {photography_mode}",
            f"Orientation: {orientation}"
        ]
        
        for measurement in measurements:
            cv2.putText(overlay, measurement, (20, y_start), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_start += line_height
        
        # Joint detection section
        y_start += 10
        cv2.putText(overlay, "JOINT DETECTION:", (10, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_start += line_height + 5
        
        if detected_joints:
            for joint, detected in detected_joints.items():
                status = "✓" if detected else "✗"
                color = (0, 255, 0) if detected else (0, 0, 255)
                cv2.putText(overlay, f"{status} {joint.replace('_', ' ').title()}", 
                           (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                y_start += 15
        
        # Blend overlay with frame
        alpha = 0.3
        cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, frame)

    def draw_basic_info_overlay(self, frame, object_distance, photography_mode):
        """Draw basic info for camera on apply mode"""
        if object_distance:
            cv2.putText(frame, f"Distance: {object_distance:.2f} ft", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            mode_color = (0, 255, 0) if photography_mode == "Portrait" else (0, 0, 255) if photography_mode == "Too Close" else (255, 0, 0)
            cv2.putText(frame, f"Mode: {photography_mode}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        else:
            cv2.putText(frame, "Person not detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def draw_info_overlay(self, frame, detected_joints, pixel_height, object_distance, 
                         photography_mode, orientation, ground_height):
        """Draw comprehensive information overlay"""
        y_offset = 30
        line_height = 20
        
        # Basic measurements
        if pixel_height and object_distance:
            cv2.putText(frame, f"Object Distance: {object_distance:.2f} ft", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += line_height
            
            cv2.putText(frame, f"Pixel Height: {pixel_height:.2f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += line_height
        
        # Ground height estimation
        if ground_height:
            cv2.putText(frame, f"Est. Ground Height: {ground_height:.2f} ft", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += line_height
        
        # Photography mode and orientation
        mode_color = (0, 255, 0) if photography_mode == "Portrait" else (255, 0, 0) if photography_mode == "Too Close" else (0, 0, 255)
        cv2.putText(frame, f"Mode: {photography_mode}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Orientation: {orientation}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Joint detection status
        if detected_joints:
            detected_count = sum(detected_joints.values())
            total_joints = len(detected_joints)
            cv2.putText(frame, f"Joints Detected: {detected_count}/{total_joints}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += line_height
        
        # Controls
        cv2.putText(frame, "Controls: 'c' to capture, 'q' to quit", 
                   (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Capture status
        if object_distance and object_distance < self.MIN_DISTANCE:
            cv2.putText(frame, "TOO CLOSE - Move back to capture", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        elif not detected_joints:
            cv2.putText(frame, "NO PERSON DETECTED", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    def advanced_capture_navigation(self, frame, outline_frame, detected_joints, distance, 
                                  photography_mode, orientation, ground_height):
        """Enhanced capture navigation with mode switching"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Create detailed info overlay for saved images
        info_frame = frame.copy()
        
        # Add detailed analysis to the saved image
        y_pos = 30
        details = [
            f"Timestamp: {timestamp}",
            f"Object Distance: {distance:.2f} ft",
            f"Photography Mode: {photography_mode}",
            f"Orientation: {orientation}",
            f"Ground Height Est: {ground_height:.2f} ft" if ground_height else "Ground Height: Not calculated"
        ]
        
        for detail in details:
            cv2.putText(info_frame, detail, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_pos += 15
        
        # Joint detection details
        y_pos += 10
        cv2.putText(info_frame, "Joint Detection Status:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        y_pos += 15
        
        for joint, detected in detected_joints.items():
            status = "✓" if detected else "✗"
            color = (0, 255, 0) if detected else (0, 0, 255)
            cv2.putText(info_frame, f"{status} {joint.replace('_', ' ').title()}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            y_pos += 12
        
        # Navigation views with mode names matching UI design
        images = [
            ("Image View Mode", frame.copy()),
            ("Outline View Mode", outline_frame.copy()),
            ("Detecting & Calculations", self.create_detecting_calculations_view(
                frame.copy(), detected_joints, 
                self.last_measurements['pixel_height'] if self.last_measurements else None,
                distance, photography_mode, orientation, ground_height)),
            ("Detailed Analysis", info_frame)
        ]
        
        idx = 0
        while True:
            title, img = images[idx]
            
            # Add mode indicator to each view
            cv2.putText(img, f"Captured: {title}", (10, img.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            display_title = f"{title} - Navigation (n/p: switch, s: save, ESC: exit)"
            cv2.imshow(display_title, img)
            
            k = cv2.waitKey(0) & 0xFF
            if k == ord('n'):  # Next view
                idx = (idx + 1) % len(images)
            elif k == ord('p'):  # Previous view
                idx = (idx - 1) % len(images)
            elif k == ord('s'):  # Save current view
                filename = f"{title.replace(' ', '_').lower()}_{photography_mode.lower()}_{timestamp}.jpg"
                cv2.imwrite(filename, img)
                print(f"{title} saved: {filename}")
                
                # Also save analysis data to text file for detailed analysis
                if title == "Detailed Analysis":
                    txt_filename = f"analysis_{photography_mode.lower()}_{timestamp}.txt"
                    with open(txt_filename, 'w') as f:
                        f.write(f"Photography Analysis Report\n")
                        f.write(f"{'='*40}\n")
                        for detail in details:
                            f.write(f"{detail}\n")
                        f.write(f"\nJoint Detection Status:\n")
                        for joint, detected in detected_joints.items():
                            status = "Detected" if detected else "Not Detected"
                            f.write(f"- {joint.replace('_', ' ').title()}: {status}\n")
                    print(f"Analysis data saved: {txt_filename}")
                    
            elif k == 27:  # ESC key
                cv2.destroyAllWindows()
                # Return to camera mode
                self.current_mode = "camera_on_apply"
                break

    def capture_and_navigate(self, frame, outline_frame, detected_joints, distance, 
                           photography_mode, orientation, ground_height):
        """Enhanced capture with detailed information"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Create detailed info overlay for saved images
        info_frame = frame.copy()
        
        # Add detailed analysis to the saved image
        y_pos = 30
        details = [
            f"Timestamp: {timestamp}",
            f"Object Distance: {distance:.2f} ft",
            f"Photography Mode: {photography_mode}",
            f"Orientation: {orientation}",
            f"Ground Height Est: {ground_height:.2f} ft" if ground_height else "Ground Height: Not calculated"
        ]
        
        for detail in details:
            cv2.putText(info_frame, detail, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_pos += 15
        
        # Joint detection details
        y_pos += 10
        cv2.putText(info_frame, "Joint Detection Status:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        y_pos += 15
        
        for joint, detected in detected_joints.items():
            status = "✓" if detected else "✗"
            color = (0, 255, 0) if detected else (0, 0, 255)
            cv2.putText(info_frame, f"{status} {joint.replace('_', ' ').title()}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            y_pos += 12
        
        images = [
            ("Image View", frame.copy()),
            ("Outline View", outline_frame.copy()),
            ("Detailed Analysis", info_frame)
        ]
        
        idx = 0
        while True:
            title, img = images[idx]
            display_title = f"{title} - Navigation (n/p: switch, s: save, ESC: exit)"
            cv2.imshow(display_title, img)
            
            k = cv2.waitKey(0) & 0xFF
            if k == ord('n'):  # Next image
                idx = (idx + 1) % len(images)
            elif k == ord('p'):  # Previous image
                idx = (idx - 1) % len(images)
            elif k == ord('s'):  # Save current image
                filename = f"{title.replace(' ', '_').lower()}_{photography_mode.lower()}_{timestamp}.jpg"
                cv2.imwrite(filename, img)
                print(f"{title} saved: {filename}")
                
                # Also save analysis data to text file
                if title == "Detailed Analysis":
                    txt_filename = f"analysis_{photography_mode.lower()}_{timestamp}.txt"
                    with open(txt_filename, 'w') as f:
                        f.write(f"Photography Analysis Report\n")
                        f.write(f"{'='*40}\n")
                        for detail in details:
                            f.write(f"{detail}\n")
                        f.write(f"\nJoint Detection Status:\n")
                        for joint, detected in detected_joints.items():
                            status = "Detected" if detected else "Not Detected"
                            f.write(f"- {joint.replace('_', ' ').title()}: {status}\n")
                    print(f"Analysis data saved: {txt_filename}")
                    
            elif k == 27:  # ESC key
                cv2.destroyAllWindows()
                break


# ------------------------------
# Main
# ------------------------------
def main():
    assistant = PhotographyAssistant()
    print("Enhanced Photography Assistant - Simple Key Controls")
    print("="*60)
    print("Features:")
    print("- Enhanced ground height calculation using geometric analysis")
    print("- Person detection with all joint tracking")
    print("- Object distance calculation") 
    print("- Portrait/Normal photography classification")
    print("- Landscape/Portrait orientation detection")
    print("- Minimum 1.5ft distance requirement for capture")
    print("\nSimple Key Controls:")
    print("- 'c' = Capture current frame")
    print("- 'o' = Show outline view (captured or live)")
    print("- 'i' = Show image view (captured or live)")
    print("- 'a' = Return to live camera view")
    print("- 'q' = Quit application")
    print("\nHow it works:")
    print("1. Start in live camera mode with pose detection")
    print("2. Press 'c' to capture when person detected & >1.5ft away")
    print("3. Use 'o', 'i', 'a' to switch between views")
    print("4. Ground height calculated using camera angle and person proportions")
    print("="*60)
    assistant.run_camera_mode()

if __name__ == "__main__":
    main()