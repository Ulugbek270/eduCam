import requests
import face_recognition
import numpy as np
from envin import embed

# Camera config
ip = "192.168.0.64"
user = "python"
password = "test12345!"

url = f"http://{user}:{password}@{ip}/ISAPI/Streaming/channels/101/picture"

# ==== Step 1:  Fetch Snapshot ====
response = requests.get(url, stream=True)

if response.status_code != 200:
    print(f"❌ Failed to get snapshot. Status: {response.status_code}")
    exit()

snapshot_path = "snapshot.jpg"
with open(snapshot_path, "wb") as f:
    f.write(response.content)
print("📸 Snapshot saved.")

# ==== Step 2: Detect Face ====
image = face_recognition.load_image_file(snapshot_path)
locations = face_recognition.face_locations(image)

if not locations:
    print("❌ No faces detected.")
    exit()

encodings = face_recognition.face_encodings(image, locations)

# ==== Step 3: Compare Each Face ====
for i, encoding in enumerate(encodings):
    distance = np.linalg.norm(encoding - embed)
    print(f"Face {i+1}: Distance = {distance:.2f}")

    if distance < 0.55:
        print("✅ Face recognized!")
    else:
        print("❌ Face not recognized.")
