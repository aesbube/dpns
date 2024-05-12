import numpy as np
import cv2


def contrast_stretching(img, params):
    for i in range(len(params) - 1):
        x0, y0 = params[i]
        x1, y1 = params[i + 1]

        if x0 >= img >= 0 == i:
            return (y0 / x0) * img
        elif x0 < img <= x1:
            return ((y1 - y0) / (x1 - x0)) * (img - x0) + y0
        elif i == len(params) - 2:
            return ((255 - y1) / (255 - x1)) * (img - x1) + y1
    return img


params_input = input("Enter params as comma-separated tuples (e.g., (50, 30), (150, 200), (255, 255)): ")

params = eval(params_input)

image = cv2.imread('image1.png', 1)

b, g, r = cv2.split(image)

pixel_val_vec = np.vectorize(lambda x: contrast_stretching(x, params))

contrast_b = pixel_val_vec(b)
contrast_g = pixel_val_vec(g)
contrast_r = pixel_val_vec(r)

stretched = cv2.merge((contrast_b, contrast_g, contrast_r))

stretched = np.array(stretched, dtype=np.uint8)

cv2.imshow('original_image', image)
cv2.imshow('modified_image', stretched)
cv2.waitKey(0)
cv2.destroyAllWindows()
