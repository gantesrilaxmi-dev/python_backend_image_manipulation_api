import os
import cv2
import numpy as np
import tensorflow as tf
import math
import time
from collections import deque
import statistics
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class PoseReplicationSystem:
    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        # Camera calibration parameters
        self.focal_length = 800
        self.confidence_threshold = 0.3
        
        # Distance thresholds for image classification
        self.PORTRAIT_MIN_DISTANCE = 45.72  # 1.5 feet in cm
        self.PORTRAIT_MAX_DISTANCE = 167.64  # 5.5 feet in cm
        self.MINIMUM_WORKING_DISTANCE = 45.72  # 1.5 feet in cm
        
        # Joint mapping
        self.joint_mapping = {
            0: 'face', 5: 'left_shoulder', 6: 'right_shoulder', 
            7: 'left_elbow', 8: 'right_elbow', 9: 'left_palm', 10: 'right_palm',
            11: 'left_hip', 12: 'right_hip', 13: 'left_knee', 14: 'right_knee', 
            15: 'left_foot', 16: 'right_foot'
        }
        
        self.required_joints = [
            'face', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_palm', 'right_palm', 'left_hip', 'right_hip', 
            'left_knee', 'right_knee', 'left_foot', 'right_foot'
        ]
        
        # Smoothing buffers
        self.distance_buffer = deque(maxlen=10)
        self.height_buffer = deque(maxlen=10)
        
        # Reference pose storage
        self.reference_pose = None
        self.current_mode = "capture_reference"  # capture_reference, replicate_pose
        
        self.load_model()
    
    def load_model(self):
        """Load the TensorFlow Lite pose estimation model"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"✅ Model loaded successfully from: {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.interpreter = None

    def preprocess_frame(self, frame):
        """Preprocess frame for model input - handles both landscape and portrait"""
        input_height = self.input_details[0]['shape'][1]
        input_width = self.input_details[0]['shape'][2]
        
        # Resize maintaining aspect ratio
        resized_frame = cv2.resize(frame, (input_width, input_height))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        if self.input_details[0]['dtype'] == np.float32:
            processed_frame = rgb_frame.astype(np.float32) / 255.0
        else:
            processed_frame = rgb_frame.astype(np.uint8)
            
        return np.expand_dims(processed_frame, axis=0)

    def run_inference(self, frame):
        """Run pose estimation inference"""
        if self.interpreter is None:
            return None
            
        input_tensor = self.preprocess_frame(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        keypoints = np.squeeze(self.interpreter.get_tensor(self.output_details[0]['index']))
        
        if keypoints.shape[1] < 3:
            confidence_column = np.ones((keypoints.shape[0], 1), dtype=np.float32)
            keypoints = np.hstack([keypoints, confidence_column])
            
        return keypoints

    def detect_person(self, keypoints):
        """Detect if a person is present in the frame"""
        if keypoints is None:
            return False
            
        high_conf_points = np.sum(keypoints[:, 2] > self.confidence_threshold)
        essential_joints = [0, 5, 6, 11, 12]  # face, shoulders, hips
        essential_detected = sum(1 for idx in essential_joints 
                               if keypoints[idx, 2] > self.confidence_threshold)
        
        return high_conf_points >= 5 and essential_detected >= 2

    def calculate_object_distance(self, keypoints, frame_shape, person_height_cm):
        """Calculate distance using provided person height"""
        if keypoints is None:
            return None
            
        frame_height = frame_shape[0]
        confident_keypoints = keypoints[keypoints[:, 2] > self.confidence_threshold]
        
        if len(confident_keypoints) < 3:
            return None
        
        y_coords = confident_keypoints[:, 0]
        person_height_pixels = (np.max(y_coords) - np.min(y_coords)) * frame_height
        
        if person_height_pixels <= 0:
            return None
        
        # Use actual person height instead of average
        distance = (person_height_cm * self.focal_length) / person_height_pixels
        
        self.distance_buffer.append(distance)
        smoothed_distance = statistics.mean(self.distance_buffer)
        
        return smoothed_distance

    def classify_image_type(self, distance):
        """Classify image as portrait or normal photography"""
        if distance is None:
            return "unknown"
        
        if self.PORTRAIT_MIN_DISTANCE <= distance <= self.PORTRAIT_MAX_DISTANCE:
            return "portrait"
        elif distance > self.PORTRAIT_MAX_DISTANCE:
            return "normal_photography"
        else:
            return "too_close"

    def is_valid_working_distance(self, distance):
        """Check if distance is valid for algorithm to work"""
        return distance is not None and distance >= self.MINIMUM_WORKING_DISTANCE

    def detect_joints(self, keypoints):
        """Detect all required joints"""
        detected_joints = {}
        
        if keypoints is None:
            for joint_name in self.required_joints:
                detected_joints[joint_name] = {
                    'detected': False,
                    'confidence': 0.0,
                    'coordinates': None
                }
            return detected_joints
        
        for idx, joint_name in self.joint_mapping.items():
            if joint_name in self.required_joints:
                y, x, conf = keypoints[idx]
                is_detected = conf > self.confidence_threshold
                
                detected_joints[joint_name] = {
                    'detected': is_detected,
                    'confidence': float(conf),
                    'coordinates': (float(x), float(y)) if is_detected else None
                }
        
        return detected_joints

    def estimate_ground_height(self, keypoints, frame_shape, object_distance):
        """Estimate camera height from ground"""
        if keypoints is None or object_distance is None:
            return None
            
        frame_height, frame_width = frame_shape[:2]
        
        foot_positions = []
        foot_indices = [15, 16]
        
        for idx in foot_indices:
            if keypoints[idx, 2] > self.confidence_threshold:
                foot_positions.append(keypoints[idx, 0])
        
        if not foot_positions:
            hip_indices = [11, 12]
            for idx in hip_indices:
                if keypoints[idx, 2] > self.confidence_threshold:
                    estimated_foot_y = keypoints[idx, 0] + 0.3
                    foot_positions.append(estimated_foot_y)
        
        if not foot_positions:
            return None
        
        avg_foot_y_normalized = np.mean(foot_positions)
        foot_pixel_y = avg_foot_y_normalized * frame_height
        camera_center_y = frame_height / 2
        vertical_angle = math.atan((foot_pixel_y - camera_center_y) / self.focal_length)
        height_difference = object_distance * math.tan(vertical_angle)
        ground_height = 140 + height_difference  # Default camera height
        ground_height = max(50, min(300, ground_height))
        
        self.height_buffer.append(ground_height)
        smoothed_height = statistics.mean(self.height_buffer)
        
        return smoothed_height

    def save_reference_pose(self, keypoints, object_distance, ground_height, person_height, detected_joints, frame_orientation):
        """Save reference pose data"""
        self.reference_pose = {
            'keypoints': keypoints.tolist() if keypoints is not None else None,
            'object_distance': object_distance,
            'ground_height': ground_height,
            'person_height': person_height,
            'detected_joints': detected_joints,
            'frame_orientation': frame_orientation,
            'image_type': self.classify_image_type(object_distance),
            'timestamp': time.strftime("%Y%m%d_%H%M%S")
        }
        
        # Save to file
        filename = f"reference_pose_{self.reference_pose['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(self.reference_pose, f, indent=2)
        
        print(f"📝 Reference pose saved to {filename}")
        return True

    def calculate_recommended_distance(self, reference_height, current_height, reference_distance):
        """Calculate recommended camera distance based on height difference"""
        if not all([reference_height, current_height, reference_distance]):
            return None
        
        # Proportional scaling: if person is taller, camera should be farther
        # Distance ratio = Height ratio
        height_ratio = current_height / reference_height
        recommended_distance = reference_distance * height_ratio
        
        return recommended_distance

    def get_distance_guidance(self, current_distance, recommended_distance):
        """Provide guidance on camera positioning"""
        if not all([current_distance, recommended_distance]):
            return "Unable to calculate guidance"
        
        distance_diff = current_distance - recommended_distance
        tolerance = 5.0  # cm tolerance
        
        if abs(distance_diff) <= tolerance:
            return "✅ Perfect position!"
        elif distance_diff > tolerance:
            return f"📷 Move camera {distance_diff:.1f}cm closer to person"
        else:
            return f"📷 Move camera {abs(distance_diff):.1f}cm away from person"

    def visualize_pose_comparison(self, frame, keypoints, reference_keypoints=None):
        """Visualize current pose with optional reference overlay"""
        if keypoints is None:
            return frame
            
        frame_copy = frame.copy()
        height, width = frame.shape[:2]
        
        # Colors for current pose
        current_colors = {
            0: (0, 255, 0),      # face - green
            5: (0, 255, 0),      # shoulders - green
            6: (0, 255, 0),
            7: (255, 0, 0),      # elbows - blue
            8: (255, 0, 0),
            9: (0, 255, 255),    # palms - yellow
            10: (0, 255, 255),
            11: (255, 0, 255),   # hips - magenta
            12: (255, 0, 255),
            13: (255, 255, 0),   # knees - cyan
            14: (255, 255, 0),
            15: (255, 255, 255), # feet - white
            16: (255, 255, 255)
        }
        
        # Draw current pose
        for i, (y, x, conf) in enumerate(keypoints):
            if conf > self.confidence_threshold:
                px, py = int(x * width), int(y * height)
                px, py = max(0, min(width-1, px)), max(0, min(height-1, py))
                
                cv2.circle(frame_copy, (px, py), 6, current_colors.get(i, (255, 255, 255)), -1)
                cv2.circle(frame_copy, (px, py), 6, (0, 0, 0), 2)
        
        # Draw reference pose if available (in red/semi-transparent)
        if reference_keypoints is not None:
            for i, (y, x, conf) in enumerate(reference_keypoints):
                if conf > self.confidence_threshold:
                    px, py = int(x * width), int(y * height)
                    px, py = max(0, min(width-1, px)), max(0, min(height-1, py))
                    
                    # Draw reference points in red with transparency effect
                    cv2.circle(frame_copy, (px, py), 4, (0, 0, 255), 2)  # Red outline
        
        # Draw skeleton connections for current pose
        connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
            (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        for start_idx, end_idx in connections:
            if (keypoints[start_idx, 2] > self.confidence_threshold and 
                keypoints[end_idx, 2] > self.confidence_threshold):
                
                start_x = int(keypoints[start_idx, 1] * width)
                start_y = int(keypoints[start_idx, 0] * height)
                end_x = int(keypoints[end_idx, 1] * width)
                end_y = int(keypoints[end_idx, 0] * height)
                
                cv2.line(frame_copy, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        return frame_copy

    def draw_enhanced_results(self, frame, person_detected, object_distance, ground_height, 
                             person_height, image_type, distance_guidance=None):
        """Draw comprehensive results on frame"""
        y_offset = 30
        line_height = 25
        
        # Mode indicator
        mode_color = (0, 255, 255) if self.current_mode == "capture_reference" else (255, 0, 255)
        mode_text = "📸 REFERENCE MODE" if self.current_mode == "capture_reference" else "🎯 REPLICATION MODE"
        cv2.putText(frame, mode_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        y_offset += line_height
        
        # Person detection
        status_color = (0, 255, 0) if person_detected else (0, 0, 255)
        status_text = "✓ Person: DETECTED" if person_detected else "✗ Person: NOT DETECTED"
        cv2.putText(frame, status_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        y_offset += line_height
        
        if person_detected:
            # Person height
            if person_height:
                height_feet = person_height / 30.48  # Convert cm to feet
                cv2.putText(frame, f"👤 Height: {person_height:.0f}cm ({height_feet:.1f}ft)", 
                          (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += line_height
            
            # Distance and validity
            if object_distance:
                distance_feet = object_distance / 30.48
                valid = self.is_valid_working_distance(object_distance)
                distance_color = (0, 255, 0) if valid else (0, 0, 255)
                
                cv2.putText(frame, f"📏 Distance: {object_distance:.1f}cm ({distance_feet:.1f}ft)", 
                          (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, distance_color, 2)
                y_offset += line_height
                
                # Image type
                type_colors = {"portrait": (0, 255, 255), "normal_photography": (255, 255, 0), 
                             "too_close": (0, 0, 255), "unknown": (128, 128, 128)}
                cv2.putText(frame, f"📷 Type: {image_type.replace('_', ' ').title()}", 
                          (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_colors.get(image_type, (255, 255, 255)), 2)
                y_offset += line_height
                
                # Working status
                if valid:
                    cv2.putText(frame, "✅ Algorithm Active", (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"❌ Too Close (Min: {self.MINIMUM_WORKING_DISTANCE:.0f}cm)", 
                              (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += line_height
            
            # Ground height
            if ground_height:
                cv2.putText(frame, f"📐 Ground Height: {ground_height:.1f}cm", 
                          (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += line_height
            
            # Distance guidance for replication mode
            if distance_guidance and self.current_mode == "replicate_pose":
                guidance_color = (0, 255, 0) if "Perfect" in distance_guidance else (255, 255, 0)
                cv2.putText(frame, distance_guidance, (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, guidance_color, 2)

    def get_frame_orientation(self, frame_shape):
        """Determine if frame is landscape or portrait"""
        height, width = frame_shape[:2]
        return "portrait" if height > width else "landscape"

    def run_camera_detection(self):
        """Main camera detection loop with pose replication system"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("🎯 Pose Replication System Started!")
        print("Controls:")
        print("  'r' - Capture reference pose (Person A)")
        print("  'm' - Switch to replication mode")
        print("  'h' - Set person height (for current person)")
        print("  'c' - Capture current frame")
        print("  'l' - Load reference pose from file")
        print("  'q' - Quit")
        
        current_person_height = None  # Will be set by user
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Get frame orientation
            frame_orientation = self.get_frame_orientation(frame.shape)
            
            # Run pose detection
            keypoints = self.run_inference(frame)
            person_detected = self.detect_person(keypoints)
            
            object_distance = None
            ground_height = None
            detected_joints = {}
            image_type = "unknown"
            distance_guidance = None
            
            if person_detected and current_person_height:
                object_distance = self.calculate_object_distance(keypoints, frame.shape, current_person_height)
                image_type = self.classify_image_type(object_distance)
                
                if self.is_valid_working_distance(object_distance):
                    detected_joints = self.detect_joints(keypoints)
                    ground_height = self.estimate_ground_height(keypoints, frame.shape, object_distance)
                    
                    # Calculate guidance for replication mode
                    if (self.current_mode == "replicate_pose" and self.reference_pose and 
                        self.reference_pose['person_height']):
                        recommended_distance = self.calculate_recommended_distance(
                            self.reference_pose['person_height'], current_person_height, 
                            self.reference_pose['object_distance'])
                        distance_guidance = self.get_distance_guidance(object_distance, recommended_distance)
            
            # Visualize pose
            if person_detected and keypoints is not None:
                reference_keypoints = None
                if self.current_mode == "replicate_pose" and self.reference_pose:
                    reference_keypoints = np.array(self.reference_pose['keypoints']) if self.reference_pose['keypoints'] else None
                
                display_frame = self.visualize_pose_comparison(frame, keypoints, reference_keypoints)
            else:
                display_frame = frame.copy()
            
            # Draw results
            self.draw_enhanced_results(display_frame, person_detected, object_distance, 
                                     ground_height, current_person_height, image_type, distance_guidance)
            
            cv2.imshow("Pose Replication System", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Capture reference pose
                if person_detected and current_person_height and self.is_valid_working_distance(object_distance):
                    success = self.save_reference_pose(keypoints, object_distance, ground_height, 
                                                     current_person_height, detected_joints, frame_orientation)
                    if success:
                        print(f"✅ Reference pose captured! Person A: {current_person_height}cm, Distance: {object_distance:.1f}cm")
                        self.current_mode = "replicate_pose"
                else:
                    print("❌ Cannot capture reference: ensure person is detected, height is set, and distance >= 1.5ft")
            
            elif key == ord('m'):
                # Switch modes
                if self.current_mode == "capture_reference":
                    if self.reference_pose:
                        self.current_mode = "replicate_pose"
                        print("🎯 Switched to replication mode")
                    else:
                        print("❌ No reference pose captured yet")
                else:
                    self.current_mode = "capture_reference"
                    print("📸 Switched to reference capture mode")
            
            elif key == ord('h'):
                # Set person height
                try:
                    height_input = input("\n👤 Enter person height in feet (e.g., 5.5): ")
                    height_feet = float(height_input)
                    current_person_height = height_feet * 30.48  # Convert to cm
                    print(f"✅ Person height set to {height_feet}ft ({current_person_height:.1f}cm)")
                except ValueError:
                    print("❌ Invalid height format. Please enter a number like 5.5")
            
            elif key == ord('c'):
                # Capture current frame
                if person_detected and current_person_height:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"pose_capture_{timestamp}.png"
                    cv2.imwrite(filename, display_frame)
                    print(f"📸 Frame captured: {filename}")
                    
                    # Print detailed results
                    distance_str = f"{object_distance:.1f}cm" if object_distance else "N/A"
                    print(f"📊 Results - Height: {current_person_height:.1f}cm, Distance: {distance_str}")
                    print(f"   Image Type: {image_type}, Orientation: {frame_orientation}")
                    if distance_guidance:
                        print(f"   Guidance: {distance_guidance}")
            
            elif key == ord('l'):
                # Load reference pose from file
                try:
                    filename = input("\n📁 Enter reference pose filename: ")
                    with open(filename, 'r') as f:
                        self.reference_pose = json.load(f)
                    print(f"✅ Reference pose loaded from {filename}")
                    self.current_mode = "replicate_pose"
                except FileNotFoundError:
                    print("❌ File not found")
                except json.JSONDecodeError:
                    print("❌ Invalid JSON file")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Pose replication system stopped")


def main():
    model_path = r"C:\Users\gante\OneDrive\Desktop\cam_web\models\pose_estimation.tflite"
    pose_system = PoseReplicationSystem(model_path)
    pose_system.run_camera_detection()

if __name__ == "__main__":
    main()