import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths to the config file and weights file
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

# Load the image
image = cv2.imread("C:/Users/91939/Downloads/birdshigh.jpg")
height, width, channels = image.shape

# Resize the image to 416x416 as YOLOv3 expects this size
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Set the blob as input to the network
net.setInput(blob)

# Get the predictions from the network
outs = net.forward(output_layers)

# Initialize lists to store detection data
class_ids = []
confidences = []
boxes = []

# Loop over each of the detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.5:  # Only consider detections with confidence > 0.5
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

# Draw bounding boxes on the image
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Convert the image to RGB (OpenCV uses BGR by default, matplotlib uses RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image with matplotlib
plt.imshow(image_rgb)
plt.axis("off")  # Hide axes
plt.show()

# Save the result
cv2.imwrite("output.jpg", image)