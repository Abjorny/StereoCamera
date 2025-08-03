import cv2

cap = cv2.VideoCapture(1)
# Установим side-by-side разрешение
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break

    # Разделим кадр пополам
    height, width = frame.shape[:2]
    left = frame[:, :width // 2]
    right = frame[:, width // 2:]

    cv2.imshow("Left", left)
    cv2.imshow("Right", right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
