from imutils.video import VideoStream, FPS
import numpy as np
import imutils
import time
import cv2

# File paths for the Caffe model
prototxt = r"D:\Codes\Random_Project\Object_Detection\Real_Time_Object_Dectation\MobileNetSSD_deploy.prototxt.txt"
model = r"D:\Codes\Random_Project\Object_Detection\Real_Time_Object_Dectation\MobileNetSSD_deploy.caffemodel"
confidence_threshold = 0.2

# Define the class labels MobileNet SSD was trained to detect
CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Assign random colors to each class label
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the serialized model
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Start the FPS counter
fps = FPS().start()

# Loop over frames from the video stream
while True:
    # Grab the frame from the video stream
    frame = vs.read()

    # Handle cases where the frame is None
    if frame is None:
        print("[ERROR] Frame is None. Exiting...")
        break

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]

    # Preprocess the frame (create a blob)
    blob = cv2.dnn.blobFromImage(frame, 1 / 127.5, (300, 300), 127.5, swapRB=True)
    net.setInput(blob)
    predictions = net.forward()

    # Loop over the predictions
    for i in np.arange(0, predictions.shape[2]):
        # Extract confidence
        confidence = predictions[0, 0, i, 2]

        # Filter weak predictions
        if confidence > confidence_threshold:
            # Extract the index of the class label and bounding box coordinates
            idx = int(predictions[0, 0, i, 1])
            box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and label on the frame
            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Show the output frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # Update the FPS counter
    fps.update()

# Stop the FPS counter and display FPS information
fps.stop()
print(f"[INFO] Elapsed time: {fps.elapsed():.2f}")
print(f"[INFO] Approximate FPS: {fps.fps():.2f}")

# Cleanup: Destroy all windows and stop the video stream
cv2.destroyAllWindows()
vs.stop()