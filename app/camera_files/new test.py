import requests
import face_recognition
import numpy as np
import cv2
import mediapipe as mp
from envin import embed  # Assuming embed is your reference face encoding

# === Config ===
ip = "192.168.0.64"
user = "python"
password = "test12345!"
snapshot_path = "snapshot.jpg"
url = f"http://{user}:{password}@{ip}/ISAPI/Streaming/channels/101/picture"

# === Step 1: Fetch Snapshot ===
response = requests.get(url, stream=True)
if response.status_code != 200:
    print(f"❌ Failed to get snapshot. Status: {response.status_code}")
    exit()

with open(snapshot_path, "wb") as f:
    f.write(response.content)
print("📸 Snapshot saved.")

# === Step 2: Face Detection & Recognition ===
image = face_recognition.load_image_file(snapshot_path)
face_locations = face_recognition.face_locations(image)

if not face_locations:
    print("❌ No faces detected.")
else:
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for i, encoding in enumerate(face_encodings):
        distance = np.linalg.norm(encoding - embed)
        print(f"Face {i + 1}: Distance = {distance:.2f}")
        if distance < 0.55:
            print("✅ Face recognized!")
        else:
            print("❌ Face not recognized.")

# === Step 3: Sitting Detection using MediaPipe ===
image_bgr = cv2.imread(snapshot_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=True)
results = pose.process(image_rgb)

sitting_count = 0
h, w, _ = image_rgb.shape

if results.pose_landmarks:
    # Extract relevant landmarks
    landmarks = results.pose_landmarks.landmark
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

    # Convert to pixel Y coordinates
    l_hip_y = left_hip.y * h
    r_hip_y = right_hip.y * h
    l_knee_y = left_knee.y * h
    r_knee_y = right_knee.y * h

    # Heuristic: If hips are close to knees vertically, assume sitting
    if abs(l_hip_y - l_knee_y) < 60 and abs(r_hip_y - r_knee_y) < 60:
        sitting_count = 1

    mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

print(f"🪑 People sitting: {sitting_count}")

# === Step 4: Show Annotated Image ===
cv2.imshow("Snapshot Analysis", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
