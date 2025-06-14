import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained Keras model
model = load_model("mnist_cnn_model.h5")

# Set up camera
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

def get_img_contour_thresh(img):
    """Preprocesses the frame to extract the digit using contour detection."""
    x, y, w, h = 0, 0, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi_thresh = thresh[y:y + h, x:x + w]
    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, roi_thresh

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ROI box
    x, y, w, h = 0, 0, 300, 300
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Get contours and threshold image
    contours, thresh = get_img_contour_thresh(frame)
    digit_pred = ''

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) > 2500:
            x1, y1, w1, h1 = cv2.boundingRect(largest_contour)
            digit_img = thresh[y1:y1 + h1, x1:x1 + w1]
            digit_img = cv2.resize(digit_img, (28, 28))
            digit_img = digit_img.astype("float32") / 255.0
            digit_img = digit_img.reshape(1, 28, 28, 1)

            # Prediction
            prediction = model.predict(digit_img, verbose=0)
            digit_pred = np.argmax(prediction)

    # Show prediction
    cv2.putText(frame, f"Predicted Digit: {digit_pred}", (10, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display windows
    cv2.imshow("Webcam Frame", frame)
    cv2.imshow("Threshold", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
