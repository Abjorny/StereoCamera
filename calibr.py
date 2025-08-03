import os
import cv2
import numpy as np
from stereovision.calibration import StereoCalibrator, StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError

# === НАСТРОЙКИ ===
rows = 6                  # количество строк (внутренние углы)
columns = 9               # количество столбцов (внутренние углы)
square_size = 2.5       # размер клетки (см или мм)
total_photos = 30      # количество пар для калибровки
image_size = (640, 280)   # размер изображений (если фиксирован)

# === Путь к папке с изображениями ===
image_folder = os.path.join(os.path.dirname(__file__), 'calibration_images')
print("📁 Текущая рабочая директория:", os.getcwd())
print("📷 Содержимое папки calibration_images:", os.listdir(image_folder))

# === Класс калибратора ===
calibrator = StereoCalibrator(rows, columns, square_size, image_size)

print('🔄 Начало цикла загрузки изображений...')
photo_counter = 0
successful = 0

while photo_counter < total_photos:
    photo_counter += 1
    left_path = os.path.join(image_folder, f'left_{photo_counter}.png')
    right_path = os.path.join(image_folder, f'right_{photo_counter}.png')

    if not (os.path.isfile(left_path) and os.path.isfile(right_path)):
        print(f"⚠️ Пара {photo_counter} не найдена, пропуск...")
        continue

    print(f"✅ Загрузка пары №{photo_counter}:")
    print(f"    {left_path}")
    print(f"    {right_path}")

    imgL = cv2.imread(left_path, 1)
    imgR = cv2.imread(right_path, 1)

    if imgL is None or imgR is None:
        print(f"❌ Ошибка чтения изображений в паре №{photo_counter}, пропуск")
        continue

    if imgL.shape != imgR.shape:
        imgR = cv2.resize(imgR, (imgL.shape[1], imgL.shape[0]))

    try:
        calibrator.add_corners((imgL, imgR))
        successful += 1
    except ChessboardNotFoundError:
        print(f"❌ Шахматка не найдена в паре №{photo_counter}, пропуск")


cv2.destroyAllWindows()
print(f"✅ Сбор данных завершён! Добавлено пар: {successful}/{total_photos}")

if successful < 3:
    print("❌ Недостаточно данных для калибровки. Нужно хотя бы 3 пары.")
    exit()

# === Калибровка камер ===
print("⚙️ Калибровка камер... Это может занять несколько минут.")
calibration = calibrator.calibrate_cameras()
calibration.export('stereo_calibration_result')
print("✅ Калибровка завершена. Параметры сохранены в папке stereo_calibration_result")

# === Пример ректификации последней пары ===
print("📐 Пример ректификации последней пары...")
calibration = StereoCalibration(input_folder='stereo_calibration_result')
rectified_pair = calibration.rectify((imgL, imgR))

cv2.imshow('Left Rectified', rectified_pair[0])
cv2.imshow('Right Rectified', rectified_pair[1])
cv2.imwrite("rectified_left.png", rectified_pair[0])
cv2.imwrite("rectified_right.png", rectified_pair[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
