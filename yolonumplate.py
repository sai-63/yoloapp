import cv2
import numpy as np
import pytesseract

# Set path for Tesseract executable (change if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Paths to YOLO configuration and weight files
configPath = "C:/projects/yolo/yolov3.cfg"
weightsPath = "C:/projects/yolo/yolov3.weights"
classesFile = "C:/projects/yolo/coco.names"

# Load YOLO
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open(classesFile, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the image
image = cv2.imread("C:/Users/91939/Downloads/bikenum.jpeg")
height, width, channels = image.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Initialize lists to store detection data
class_ids = []
confidences = []
boxes = []

# Loop through YOLO detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.5:  # Consider only confident detections
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression to filter out overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

# Loop over the detections and draw bounding boxes on the image
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    
    # Draw bounding box for the detected object
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # If we detect a number plate
    if label == "number_plate":  # Ensure this matches the label in your YOLO dataset
        roi = image[y:y+h, x:x+w]
        
        # Preprocess the detected number plate region for OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use pytesseract to extract the text (number plate)
        text = pytesseract.image_to_string(thresholded, config='--psm 8')  # 8 means treat the image as a single word
        
        # Display the extracted number plate text
        print(f"Detected number plate text: {text.strip()}")
        cv2.putText(image, text.strip(), (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Show the processed image
cv2.imshow("Image with Number Plates", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the image with annotations
cv2.imwrite("output_image_with_number_plates.jpg", image)