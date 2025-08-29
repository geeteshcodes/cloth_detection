import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained Fashion MNIST model
model = load_model("fashion.h5")
classes = ["T-shirt","Trouser","Pullover","Dress","Coat",
           "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Keep original frame for display
    display_frame = frame.copy()

    # Resize a copy for faster processing
    frame_resized = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Threshold and find contours
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Largest contour assumed to be clothing
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Map coordinates back to original frame size
        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 240
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        # Extract ROI from original frame (grayscale for model)
        cloth_roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

        # Resize for Fashion MNIST model
        resized = cv2.resize(cloth_roi, (28,28))
        normalized = resized / 255.0
        input_frame = np.expand_dims(normalized, axis=(0,-1))

        # Predict
        pred = model.predict(input_frame, verbose=0)
        class_index = np.argmax(pred)
        label = classes[class_index]
        confidence = np.max(pred)

        # Draw rectangle and label on display frame
        cv2.rectangle(display_frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(display_frame, f"{label} ({confidence:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Show extracted clothing region in separate window
        cv2.imshow("Clothing ROI", cloth_roi)

    # Show full-size webcam feed
    cv2.imshow("Fashion MNIST Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
