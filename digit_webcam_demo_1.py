import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mnist_cnn_model.h5")

# Start the webcam
cap = cv2.VideoCapture(0)
print("Press SPACE to predict a digit. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optional: Flip horizontally (comment if mirrored)
    # frame = cv2.flip(frame, 1)

    # Draw ROI rectangle
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    roi = gray[100:300, 100:300]

    # Show the webcam feed
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

    elif key == 32:  # SPACE
        # Preprocess the image for prediction
        roi = cv2.resize(roi, (28, 28))
        roi = cv2.GaussianBlur(roi, (5, 5), 0)

        # Invert colors to match MNIST (white digit on black)
        roi = cv2.bitwise_not(roi)

        # Apply binary threshold
        _, roi = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)

        # Normalize and reshape
        roi = roi.astype("float32") / 255.0
        roi = roi.reshape(1, 28, 28, 1)

        # Predict the digit
        prediction = model.predict(roi)
        digit = np.argmax(prediction)

        print(f"Predicted Digit: {digit}")

        # Display result on the frame
        output_frame = frame.copy()
        cv2.putText(output_frame, f"Digit: {digit}", (100, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Prediction", output_frame)
        cv2.waitKey(1500)

cap.release()
cv2.destroyAllWindows()
