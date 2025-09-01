import torch
import cv2

# Step 1: Load the pre-trained YOLOv5 model from PyTorch Hub.
# The 'yolov5s' model is a small, fast version perfect for real-time applications.
# If you have a GPU, the model will automatically use it.
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    print("YOLOv5 model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check your internet connection and ensure PyTorch is correctly installed.")
    exit()

# Step 2: Initialize the webcam.
# '0' typically refers to the default camera on your computer.
# If you have an external webcam, you might need to change this number (e.g., 1, 2, etc.).
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully.
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# The script will now wait for the user to press 'q' to quit.
print("\nWebcam initialized. Press 'q' to quit the video feed.")

# We'll use this loop to read frames from the webcam.

while True:
    # Read a frame from the webcam.
    ret, frame = cap.read()

    # If the frame was not read correctly, break the loop.
    if not ret:
        break

    # Perform object detection on the frame.
    # The model() function handles all the preprocessing and inference.
    results = model(frame)

    # The results object contains the detected objects, bounding boxes, and labels.
    # We can get the annotated image with bounding boxes drawn on it.
    annotated_frame = results.render()[0]

    # Display the annotated frame.
    cv2.imshow('YOLO Object Detection', annotated_frame)

    # Break the loop if the user presses the 'q' key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the webcam and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()