import os
import glob

import cv2
import numpy as np

def make_contours(images):
    for img in images:
        image = cv2.imread(img)
        processed_image = image.copy()

        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.GaussianBlur(processed_image, (5, 5), 0)
        _, processed_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, kernel=np.ones((5, 5)), iterations=1)
        processed_image = 255 - processed_image

        contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_image = cv2.drawContours(np.zeros(processed_image.shape, np.uint8), contours, -1, 255, 1)

        cv2.imwrite(os.path.join(output, os.path.basename(img)), 255 - final_image)


data_input = './database'
output = './result'

input_images = glob.glob(os.path.join(data_input, "*"))
os.makedirs(output, exist_ok=True)

make_contours(input_images)

# избраниот алгоритам точно ги сегментира дадените слики од database и резултатните слики се наоѓаат во result фолдерот