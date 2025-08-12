import cv2
import numpy as np

# Загрузка параметров калибровки
data = np.load("stereo_params.npz")
mtxL, distL = data['mtxL'], data['distL']
mtxR, distR = data['mtxR'], data['distR']
R1, R2 = data['R1'], data['R2']
P1, P2 = data['P1'], data['P2']
Q = data['Q']

# Инициализация камеры
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 280)

ret, frame = cap.read()
if not ret:
    print("Ошибка: не удалось получить кадр с камеры.")
    cap.release()
    exit()

h, w = frame.shape[:2]
left = frame[:, :w // 2]
right = frame[:, w // 2:]
image_size = (left.shape[1], left.shape[0])  # (320, 280)

# Карты выравнивания
map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, image_size, cv2.CV_32FC1)

# StereoSGBM конфигурация
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=48,  
    blockSize=3,
    P1=8*3*5**2,
    P2=32*3*5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=200,
    speckleRange=2
)

print("Нажмите 'q' для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    left = frame[:, :w // 2]
    right = frame[:, w // 2:]
    print(left.shape, right.shape)  # Должны быть одинаковые размеры
    # Ректификация
    rectL = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

    # Перевод в оттенки серого
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    # Вычисление карты диспаратности
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Ограничим значения и обрежем артефакты
    disparity[disparity < 0] = 0
    disparity[disparity > 128] = 128

    # Нормализация и цветовая визуализация
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # Медианная фильтрация для сглаживания карты
    disp_color = cv2.medianBlur(disp_color, 5)

    # Проверка горизонтального выравнивания
    align_check = cv2.hconcat([rectL, rectR])
    for y in range(0, align_check.shape[0], 20):
        cv2.line(align_check, (0, y), (align_check.shape[1], y), (0, 255, 0), 1)

    # Отображение
    cv2.imshow("Disparity (Color)", disp_color)
    cv2.imshow("Rectified Left", rectL)
    cv2.imshow("Rectified Right", rectR)
    cv2.imshow("Stereo Alignment Check", align_check)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
