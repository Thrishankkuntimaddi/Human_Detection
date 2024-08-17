import cv2
import numpy as np
import time


# Function to convert X-coordinate to angle (in radians)
def x_coordinate_to_angle(x_coord, reference_x):
    """
    Convert an X-coordinate to an angle (in radians) relative to a reference X-coordinate.

    Parameters:
    - x_coord (int): The X-coordinate to convert.
    - reference_x (int): The reference X-coordinate (e.g., center of image).

    Returns:
    - float: Angle in radians.
    """
    # Calculate the difference in X-coordinates
    delta_x = x_coord - reference_x

    # Assuming a simple trigonometric relationship (tan(theta) = opposite/adjacent)
    # Calculate the angle (in radians) using arctangent
    angle_rad = np.arctan2(delta_x, 1.0)  # Assuming 1.0 as the fixed distance or reference

    return angle_rad


# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the camera (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Get the width of the frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Reference X-coordinate (center of the frame)
reference_x = frame_width // 2

last_print_time = time.time()

while True:
    # Read frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally to get a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale (face detection works on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    x_coords = []  # List to store X-coordinates

    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        # Append the X-coordinate
        x_coords.append(x)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label the face
        cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Print X-coordinates and angles to the console every second
    current_time = time.time()
    if current_time - last_print_time >= 1:
        if x_coords:
            print("Details of detected faces:")
            for idx, (x, y, w, h) in enumerate(faces):
                angle_rad = x_coordinate_to_angle(x, reference_x)
                print(f"Face {idx + 1}: X = {x}, Y = {y}, Width = {w}, Height = {h}, Angle (radians) = {angle_rad:.2f}")
        else:
            print("No faces detected.")
        last_print_time = current_time

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
