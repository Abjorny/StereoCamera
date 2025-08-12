import cv2
import os

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 280)

output_dir = "calibration_images"
os.makedirs(output_dir, exist_ok=True)
counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]
    left = frame[:, :w // 2]
    right = frame[:, w // 2:]

    cv2.imshow("Left", left)
    cv2.imshow("Right", right)

    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite(f"{output_dir}/left_{counter}.png", left)
        cv2.imwrite(f"{output_dir}/right_{counter}.png", right)
        print(f"Saved pair {counter}")
        counter += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()