import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 620)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


stereo = cv2.StereoBM_create(numDisparities=96, blockSize=13)



while True:
    ret, frame = cap.read()

    height, width = frame.shape[:2]
    left_frame = frame[:, :width // 2]
    right_frame = frame[:, width // 2:]

    gray_left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    disparity = (stereo.compute(gray_left_frame, gray_right_frame)+16.)/256.

    cv2.imshow("disparity", disparity)

    # выделить
    ret, disparity_tresh = cv2.threshold(disparity, 0.1, 255, cv2.THRESH_BINARY_INV)
    disparity_tresh = disparity_tresh.astype(np.uint8)
    cv2.imshow("disparity_tresh", disparity_tresh)


    cv2.imshow("Frame", np.hstack([left_frame, right_frame]))
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
