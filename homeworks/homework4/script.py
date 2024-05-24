import os
import cv2
import numpy as np

def get_contours(img_path):
    image = cv2.imread(img_path)
    greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(greyscale_image, (5, 5), 0)
    _, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morphed_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel=np.ones((5, 5)), iterations=1)
    inverted_image = 255 - morphed_image
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_image = cv2.drawContours(np.zeros(inverted_image.shape, np.uint8), contours, -1, 255, 1)
    results_dir = './result'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    cv2.imwrite(os.path.join(results_dir, os.path.basename(img_path)), final_image)
    return contours[0] if contours else None

def main():
    similarities = {}
    input_folder = './database/'
    query_folder = './query_images/'

    query_image_name = input('Image: ')
    query_image_path = os.path.join(query_folder, query_image_name)
    query_contour = get_contours(query_image_path)
    
    if query_contour is None:
        print(f"No contours found in the query image: {query_image_name}")
        return

    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for image_path in image_files:
        contour = get_contours(image_path)
        if contour is not None:
            similarity = cv2.matchShapes(query_contour, contour, cv2.CONTOURS_MATCH_I1, 0)
            similarities[os.path.basename(image_path)] = similarity

    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1])

    for filename, similarity in sorted_similarities:
        print(f'{filename}:\t{similarity}')

if __name__ == '__main__':
    main()
