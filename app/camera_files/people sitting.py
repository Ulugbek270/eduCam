import requests
import cv2

# Camera config
ip = "192.168.0.64"
user = "python"
password = "test12345!"
url = f"http://{user}:{password}@{ip}/ISAPI/Streaming/channels/101/picture"
snapshot_path = "snapshot.jpg"

# === Step 1: Get Snapshot ===
response = requests.get(url, stream=True)
if response.status_code != 200:
    print(f"❌ Failed to get snapshot. Status: {response.status_code}")
    exit()

with open(snapshot_path, "wb") as f:
    f.write(response.content)
print("📸 Snapshot saved.")

# === Step 2: Load Model (MobileNet SSD) ===
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "mobilenet_iter_73000.caffemodel"
)

# === Step 3: Detect People ===
image = cv2.imread(snapshot_path)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
net.setInput(blob)
detections = net.forward()

person_count = 0

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    class_id = int(detections[0, 0, i, 1])
    if confidence > 0.5 and class_id == 15:  # 15 is 'person' in COCO
        person_count += 1
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

print(f"👥 People detected: {person_count}")
cv2.imshow("Detected People", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
