import os
import cv2
import numpy as np
import tensorflow as tf
import math
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class PoseDetectionSystem:
    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.focal_length = 800
        self.average_person_height = 170
        self.confidence_threshold = 0.3
        self.joint_mapping = {
            0: 'face', 5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow',
            8: 'right_elbow', 9: 'left_palm', 10: 'right_palm', 11: 'left_hip',
            12: 'right_hip', 13: 'left_knee', 14: 'right_knee', 15: 'left_foot', 16: 'right_foot'
        }
        self.required_joints = list(self.joint_mapping.values())
        self.load_model()
    
    def load_model(self):
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"Model loaded from: {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.interpreter = None

    def preprocess_frame(self, frame):
        input_height = self.input_details[0]['shape'][1]
        input_width = self.input_details[0]['shape'][2]
        resized_frame = cv2.resize(frame, (input_width, input_height))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        if self.input_details[0]['dtype'] == np.float32:
            processed_frame = rgb_frame.astype(np.float32) / 255.0
        else:
            processed_frame = rgb_frame.astype(np.uint8)
        return np.expand_dims(processed_frame, axis=0)

    def run_inference(self, frame):
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
        if keypoints is None:
            return False
        return np.sum(keypoints[:, 2] > self.confidence_threshold) >= 5

    def calculate_object_distance(self, keypoints, frame_shape):
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
        return (self.average_person_height * self.focal_length) / person_height_pixels

    def detect_joints(self, keypoints):
        detected_joints = {}
        if keypoints is None:
            for joint_name in self.required_joints:
                detected_joints[joint_name] = {'detected': False, 'confidence': 0.0, 'coordinates': None}
            return detected_joints
        for idx, joint_name in self.joint_mapping.items():
            if joint_name in self.required_joints:
                y, x, conf = keypoints[idx]
                detected_joints[joint_name] = {
                    'detected': conf > self.confidence_threshold,
                    'confidence': float(conf),
                    'coordinates': (float(x), float(y)) if conf > self.confidence_threshold else None
                }
        return detected_joints

    def estimate_ground_height(self, keypoints, frame_shape, object_distance):
        if keypoints is None or object_distance is None:
            return None
        frame_height, frame_width = frame_shape[:2]
        foot_positions = []
        for idx in [15,16]:
            if keypoints[idx,2] > self.confidence_threshold:
                foot_positions.append(keypoints[idx,0])
        if not foot_positions:
            for idx in [11,12]:
                if keypoints[idx,2] > self.confidence_threshold:
                    foot_positions.append(keypoints[idx,0] + 0.3)
        if not foot_positions:
            return None
        avg_foot_y = np.mean(foot_positions)
        foot_pixel_y = avg_foot_y * frame_height
        vertical_angle = math.atan((foot_pixel_y - frame_height/2)/self.focal_length)
        height_difference = object_distance * math.tan(vertical_angle)
        ground_height = 140 + height_difference
        return max(50, min(300, ground_height))

    def visualize_pose(self, frame, keypoints):
        if keypoints is None:
            return frame
        frame_copy = frame.copy()
        height, width = frame.shape[:2]
        colors = {0:(0,0,255),5:(0,255,0),6:(0,255,0),7:(255,0,0),8:(255,0,0),
                  9:(0,255,255),10:(0,255,255),11:(255,0,255),12:(255,0,255),
                  13:(255,255,0),14:(255,255,0),15:(255,255,255),16:(255,255,255)}
        for i,(y,x,conf) in enumerate(keypoints):
            if conf>self.confidence_threshold:
                px, py = int(x*width), int(y*height)
                px, py = max(0,min(width-1,px)), max(0,min(height-1,py))
                cv2.circle(frame_copy,(px,py),5,colors.get(i,(255,255,255)),-1)
                if i in self.joint_mapping:
                    cv2.putText(frame_copy,self.joint_mapping[i],(px+5,py-5),
                                cv2.FONT_HERSHEY_SIMPLEX,0.3,colors.get(i,(255,255,255)),1)
        connections = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),
                       (11,12),(11,13),(13,15),(12,14),(14,16)]
        for s,e in connections:
            if keypoints[s,2]>self.confidence_threshold and keypoints[e,2]>self.confidence_threshold:
                sx, sy = int(keypoints[s,1]*width), int(keypoints[s,0]*height)
                ex, ey = int(keypoints[e,1]*width), int(keypoints[e,0]*height)
                cv2.line(frame_copy,(sx,sy),(ex,ey),(0,255,0),2)
        return frame_copy

    def draw_results(self, frame, person_detected, object_distance, detected_joints, ground_height):
        y_offset, line_height = 30, 25
        status_color = (0,255,0) if person_detected else (0,0,255)
        status_text = "Person: DETECTED" if person_detected else "Person: NOT DETECTED"
        cv2.putText(frame,status_text,(10,y_offset),cv2.FONT_HERSHEY_SIMPLEX,0.7,status_color,2)
        y_offset += line_height
        if person_detected:
            if object_distance:
                cv2.putText(frame,f"Distance: {object_distance:.1f} cm",(10,y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
                y_offset += line_height
            if ground_height:
                cv2.putText(frame,f"Ground Height: {ground_height:.1f} cm",(10,y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
                y_offset += line_height
            detected_count = sum(1 for j in detected_joints.values() if j['detected'])
            total_count = len(detected_joints)
            cv2.putText(frame,f"Joints: {detected_count}/{total_count}",(10,y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

    # ------------------- Edge Detection Functions -------------------
    def get_person_mask(self, frame):
        """
        Replace this function with your actual segmentation model.
        For demo, we assume the person is the entire frame.
        """
        mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        return mask

    def get_person_edges(self, frame, mask):
        edges = cv2.Canny(mask*255, 50, 150)
        frame_edges = frame.copy()
        frame_edges[edges!=0] = [0,0,255]  # red edges
        return frame_edges

    # ----------------------------------------------------------------

    def run_camera_detection(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("Press 'c' to capture, 'o' to show last capture (with joints), 'i' for last raw image, 'a' for live, 'e' for edges, 'q' to quit")

        mode = "live"
        last_with_joints, last_raw = None, None

        while True:
            if mode in ["live","capture"]:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame")
                    break

                keypoints = self.run_inference(frame)
                person_detected = self.detect_person(keypoints)
                object_distance, detected_joints, ground_height = None, {}, None

                if person_detected:
                    object_distance = self.calculate_object_distance(keypoints, frame.shape)
                    detected_joints = self.detect_joints(keypoints)
                    ground_height = self.estimate_ground_height(keypoints, frame.shape, object_distance)
                    frame_with_pose = self.visualize_pose(frame, keypoints)
                    self.draw_results(frame_with_pose, person_detected, object_distance, detected_joints, ground_height)
                else:
                    frame_with_pose = frame.copy()

                if mode=="capture":
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    file_joints = f"pose_capture_with_joints_{timestamp}.png"
                    file_raw = f"pose_capture_raw_{timestamp}.png"
                    cv2.imwrite(file_joints, frame_with_pose)
                    cv2.imwrite(file_raw, frame)
                    last_with_joints = frame_with_pose.copy()
                    last_raw = frame.copy()
                    print(f"Captured images saved: {file_joints} and {file_raw}")
                    mode = "live"

                display_frame = frame_with_pose

            elif mode=="show_with_joints":
                if last_with_joints is None:
                    display_frame = np.zeros((480,640,3),dtype=np.uint8)
                    cv2.putText(display_frame,"No captured image with joints",(50,240),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                else:
                    display_frame = last_with_joints.copy()

            elif mode=="show_raw":
                if last_raw is None:
                    display_frame = np.zeros((480,640,3),dtype=np.uint8)
                    cv2.putText(display_frame,"No captured raw image",(50,240),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                else:
                    display_frame = last_raw.copy()

            elif mode=="show_edges":
                if last_raw is None:
                    display_frame = np.zeros((480,640,3),dtype=np.uint8)
                    cv2.putText(display_frame,"No captured raw image for edges",(50,240),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                else:
                    mask = self.get_person_mask(last_raw)
                    display_frame = self.get_person_edges(last_raw, mask)

            cv2.imshow("Pose Detection System", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key==ord('q'):
                break
            elif key==ord('c'):
                mode="capture"
            elif key==ord('o'):
                mode="show_with_joints"
            elif key==ord('i'):
                mode="show_raw"
            elif key==ord('a'):
                mode="live"
            elif key==ord('e'):
                mode="show_edges"

        cap.release()
        cv2.destroyAllWindows()


# ==========================
def main():
    model_path = r"C:\Users\gante\OneDrive\Desktop\cam_web\models\pose_estimation.tflite"
    pose_system = PoseDetectionSystem(model_path)
    pose_system.run_camera_detection()

if __name__ == "__main__":
    main()

