import cv2
import numpy as np

# Paths to the config file, weights file, and class labels file
configPath = "C:/projects/yolo/yolov3.cfg"
weightsPath = "C:/projects/yolo/yolov3.weights"
classesFile = "C:/projects/yolo/coco.names"

# Load YOLO
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the class labels (COCO dataset labels)
with open(classesFile, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open video file or webcam
cap = cv2.VideoCapture("C:/Users/91939/Downloads/cars.mp4")  # Or use 0 for webcam

# Get the frame width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object (use XVID or other codecs)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Or use 'MJPG', 'MP4V', etc.
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (width, height))  # 20 FPS

# Ensure the VideoWriter is initialized correctly
if not out.isOpened():
    print("Error: VideoWriter not opened!")
    exit()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to store detection data
    class_ids = []
    confidences = []
    boxes = []

    for layer_output in outs:  # Renamed 'out' to 'layer_output'
        for detection in layer_output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Draw bounding boxes on the frame
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Resize the frame if necessary
    frame_resized = cv2.resize(frame, (width, height))

    # Check if frame is valid before writing
    if frame_resized is not None:
        out.write(frame_resized)
    else:
        print("Frame is None, skipping write.")

    # Show the processed frame
    cv2.imshow("Frame", frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()