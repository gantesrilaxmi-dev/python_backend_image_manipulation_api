from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from photography_upload_app import PoseReplicationSystem  # Your class

app = Flask(__name__)

# Initialize the pose system
model_path = r"C:\Users\gante\OneDrive\Desktop\cam_web\models\pose_estimation.tflite"
pose_system = PoseReplicationSystem(model_path)

@app.route("/process_frame", methods=["POST"])
def process_frame():
    """
    Expects a JSON payload with:
    {
        "image": "<base64_encoded_image>",
        "person_height_ft": 5.5   # optional if height known
    }
    """
    data = request.json
    if "image" not in data:
        return jsonify({"error": "Image is required"}), 400
    
    # Decode base64 image
    image_b64 = data["image"]
    image_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Get person height if provided
    person_height_cm = None
    if "person_height_ft" in data:
        try:
            person_height_cm = float(data["person_height_ft"]) * 30.48
        except:
            return jsonify({"error": "Invalid person_height_ft"}), 400

    # Run inference
    keypoints = pose_system.run_inference(frame)
    person_detected = pose_system.detect_person(keypoints)
    frame_orientation = pose_system.get_frame_orientation(frame.shape)

    object_distance = None
    ground_height = None
    detected_joints = {}
    image_type = "unknown"

    if person_detected and person_height_cm:
        object_distance = pose_system.calculate_object_distance(keypoints, frame.shape, person_height_cm)
        image_type = pose_system.classify_image_type(object_distance)

        if pose_system.is_valid_working_distance(object_distance):
            detected_joints = pose_system.detect_joints(keypoints)
            ground_height = pose_system.estimate_ground_height(keypoints, frame.shape, object_distance)

    # Prepare JSON response
    response = {
        "person_detected": person_detected,
        "object_distance_cm": round(object_distance, 2) if object_distance else None,
        "ground_height_cm": round(ground_height, 2) if ground_height else None,
        "detected_joints": detected_joints,
        "image_type": image_type,
        "frame_orientation": frame_orientation
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
