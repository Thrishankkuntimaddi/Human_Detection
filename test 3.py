import cv2
import numpy as np
import time

net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

cap = cv2.VideoCapture(1)


if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

reference_x = frame_width // 2

last_print_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    x_coords = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
            (startX, startY, endX, endY) = box.astype("int")
            face_width = endX - startX

            x_coords.append(startX)

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f'Face: {confidence:.2f}'
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    current_time = time.time()
    if current_time - last_print_time >= 1:
        if x_coords:
            print("Details of detected faces:")
            for idx, x in enumerate(x_coords):
                angle_rad = np.arctan2(x - reference_x, 1.0)  # Calculate angle relative to reference_x
                print(f"Face {idx + 1}: X = {x}, Y = {startY}, Width = {face_width}, Height = {endY - startY}, Angle (radians) = {angle_rad:.2f}")
        else:
            print("No faces detected.")
        last_print_time = current_time

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
