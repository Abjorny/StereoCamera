import os
import cv2
import numpy as np

# === Параметры шахматной доски ===
rows = 6
columns = 9
square_size = 1.5  # В сантиметрах или миллиметрах
image_size = (640, 280)
total_photos = 30

# === Подготовка точек шахматной доски ===
objp = np.zeros((rows * columns, 3), np.float32)
objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
objp *= square_size

# === Списки для хранения точек ===
objpoints = []  # 3D точки в реальном мире
imgpointsL = []  # 2D точки на изображении левой камеры
imgpointsR = []

# === Папка с изображениями ===
image_folder = os.path.join(os.path.dirname(__file__), 'calibration_images')

print("📁 Текущая рабочая директория:", os.getcwd())
print("📷 Загрузка изображений из:", image_folder)

successful = 0
for i in range(1, total_photos + 1):
    left_path = os.path.join(image_folder, f'left_{i}.png')
    right_path = os.path.join(image_folder, f'right_{i}.png')

    if not (os.path.exists(left_path) and os.path.exists(right_path)):
        print(f"⚠️ Пропущена пара №{i} — файл не найден")
        continue

    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, (columns, rows), None)
    retR, cornersR = cv2.findChessboardCorners(grayR, (columns, rows), None)

    if retL and retR:
        objpoints.append(objp)
        corners2L = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1),
                                     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        corners2R = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1),
                                     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpointsL.append(corners2L)
        imgpointsR.append(corners2R)
        successful += 1
        print(f"✅ Успешно найдена шахматка на паре №{i}")
    else:
        print(f"❌ Шахматка не найдена на паре №{i}")

print(f"\n🔍 Найдено валидных пар: {successful}/{total_photos}")
if successful < 3:
    print("❌ Недостаточно пар для калибровки. Нужно хотя бы 3.")
    exit()

# === Калибровка одиночных камер ===
retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, image_size, None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, image_size, None, None)

# === Стереокалибровка ===
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

_, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    image_size, criteria=criteria, flags=flags
)

# === Ректификация ===
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, image_size, R, T, alpha=0
)

# === Сохранение параметров ===
np.savez("stereo_params.npz",
         mtxL=mtxL, distL=distL,
         mtxR=mtxR, distR=distR,
         R=R, T=T, E=E, F=F,
         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

print("\n✅ Параметры калибровки сохранены в stereo_params.npz")
