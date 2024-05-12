import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = './image1.jpg'
input_image = cv2.imread(image_path)

convolution_kernels = [
    np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]]),
    np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
    np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]),
    np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
    np.array([]),
    np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]),
    np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
]

processed_images = []
scale_factor = float(input('Enter the scale factor for the kernels: '))

plot_rows, plot_cols = 3, 3
fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(10, 6))
axes_list = axes.flatten()

for index, kernel in enumerate(convolution_kernels):
    if index == 4:
        continue

    processed_image = cv2.filter2D(src=input_image, ddepth=-1, kernel=kernel * scale_factor)
    processed_images.append(processed_image)
    axes_list[index].imshow(processed_image)
    axes_list[index].set_title(f'Kernel {index}')
    axes_list[index].axis('off')

final_image = np.maximum.reduce(processed_images)

axes_list[4].imshow(final_image)
axes_list[4].set_title('Combined Result')
axes_list[4].axis('off')

plt.show()
