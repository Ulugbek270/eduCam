import numpy as np
import cv2
import face_recognition
import time
from envin import rtsp_url, embed

# Your known embedding
known_embedding = np.array(embed)  # ensure it's a numpy array

# Print stream info before starting
print(f"Connecting to RTSP stream: {rtsp_url}")

# Initialize variables for frame dumping in case of errors
frame_dump_counter = 0
last_frame_dump_time = time.time()

# Open RTSP stream with optimized parameters
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

# Try setting lower resolution to help with processing speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Try to set buffer size to minimize delay
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Failed to open RTSP stream")
    exit()

print("✅ Successfully connected to RTSP stream")

# Create window
cv2.namedWindow("Hikvision Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hikvision Face Recognition", 1280, 720)

# Debug variables
frame_count = 0
success_count = 0
last_status_time = time.time()


# Function to save problematic frames for debugging
def save_problematic_frame(frame, prefix="debug"):
    global frame_dump_counter
    try:
        if frame is not None and frame.size > 0:
            filename = f"{prefix}_frame_{frame_dump_counter}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved problematic frame to {filename}")
            frame_dump_counter += 1
    except Exception as e:
        print(f"Could not save debug frame: {e}")


while True:
    try:
        # Read frame
        ret, frame = cap.read()
        frame_count += 1

        # Print status every 5 seconds
        current_time = time.time()
        if current_time - last_status_time > 5:
            print(
                f"Processed {frame_count} frames, {success_count} successful detections"
            )
            last_status_time = current_time

        # Check if frame is valid
        if not ret or frame is None or frame.size == 0:
            print("❌ Empty frame received, retrying...")
            # Try to reconnect if stream is disconnected
            cap.release()
            time.sleep(1)  # Wait before reconnecting
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            continue

        # Deep copy the frame to ensure we're not working with a reference
        frame = frame.copy()

        # Check frame properties
        if frame.dtype != np.uint8:
            print(f"⚠️ Converting {frame.dtype} to uint8...")
            frame = frame.astype(np.uint8)

        # Resize frame for faster processing (even smaller than before)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert BGR to RGB correctly
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Add debugging info (frame dimensions, type)
        if frame_count % 100 == 0:
            print(
                f"Frame shape: {rgb_small_frame.shape}, dtype: {rgb_small_frame.dtype}"
            )

        # Special case: If we're getting intermittent errors, dump frames occasionally
        if time.time() - last_frame_dump_time > 60:  # Every minute
            save_problematic_frame(rgb_small_frame, "periodic")
            last_frame_dump_time = time.time()

        # Explicitly force the correct format for face_recognition (uint8, 3 channels)
        if rgb_small_frame.dtype != np.uint8 or len(rgb_small_frame.shape) != 3:
            print("⚠️ Invalid frame format, forcing correct format...")
            if len(rgb_small_frame.shape) != 3:
                # Convert grayscale to RGB if needed
                rgb_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_GRAY2RGB)
            rgb_small_frame = rgb_small_frame.astype(np.uint8)

        # Use HOG model explicitly for face detection
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

        # Only if faces are found
        if face_locations:
            success_count += 1
            # Compute embeddings
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations
            )

            for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings
            ):
                # Compare detected face to known embedding
                distance = np.linalg.norm(face_encoding - known_embedding)

                # You might need to adjust this threshold based on your needs
                if distance < 0.6:
                    label = f"✅ Recognized! ({distance:.2f})"
                    color = (0, 255, 0)
                else:
                    label = f"❌ Unknown ({distance:.2f})"
                    color = (0, 0, 255)

                # Scale back to original frame size (4x since we used 0.25 scale)
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw box and label
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                )

                # Print successful recognition with timestamp
                print(
                    f"Face detected at {time.strftime('%H:%M:%S')}! Distance: {distance:.2f}"
                )

        # Add processing stats to the frame
        cv2.putText(
            frame,
            f"Frames: {frame_count} | Detections: {success_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    except Exception as e:
        print(f"⚠️ Error processing frame: {e}")
        # Save problematic frame for debugging
        save_problematic_frame(frame if "frame" in locals() else None, "error")
        import traceback

        traceback.print_exc()
        continue  # continue safely if anything fails

    # Show the final frame
    cv2.imshow("Hikvision Face Recognition", frame)

    # Break loop with 'q' key
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    # Save screenshot with 's' key
    elif key == ord("s"):
        cv2.imwrite("face_detection_screenshot.jpg", frame)
        print("📸 Screenshot saved!")

# Clean up
print("Closing application...")
cap.release()
cv2.destroyAllWindows()
