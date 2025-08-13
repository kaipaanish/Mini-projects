import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ---- RED detection ----
    lower_red = np.array([161, 155, 84])
    upper_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)

    # ---- GREEN detection ----
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)

    # ---- Non-white detection ----
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    non_white_mask = cv2.bitwise_not(white_mask)
    non_white = cv2.bitwise_and(frame, frame, mask=non_white_mask)

    # ---- Combine into a single window ----
    top_row = np.hstack((frame, red))
    bottom_row = np.hstack((green, non_white))
    combined = np.vstack((top_row, bottom_row))

    cv2.imshow("Original | Red | Green | Non-White", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
