import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White color range
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # Mask for white
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)

    # Invert mask to get everything except white
    non_white_mask = cv2.bitwise_not(white_mask)

    # Apply mask
    result = cv2.bitwise_and(frame, frame, mask=non_white_mask)

    cv2.imshow("Non-White Colors", result)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
