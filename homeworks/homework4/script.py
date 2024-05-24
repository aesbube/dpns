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

# Коментари:

# Image: 1202.jpg
# 10001.jpg:      0.03736500921163333
# 11721.jpg:      0.06926498079136442
# 10368.jpg:      0.09237984275444677
# 10152.jpg:      0.10729774969562808
# 11361.jpg:      0.11745409851147401
# 10939.jpg:      0.13058100387842736
# 1.jpg:  0.15420450097380028
# 11270.jpg:      0.15511368050118834
# 10492.jpg:      0.17813163249688133
# 11250.jpg:      0.18459899224172227
# 10677.jpg:      0.19840225850690762
# 11190.jpg:      0.20526028176258812
# 10002.jpg:      0.21890320563560797
# 11248.jpg:      0.23172303443167996
# 11628.jpg:      0.23939677686631888
# 10414.jpg:      0.2468322583228315
# 10479.jpg:      0.24921672259934027
# 13567.jpg:      0.39399452548266833
# 10752.jpg:      0.39938428528829095
# 10000.jpg:      0.49219025331976685
# 10379.jpg:      0.5210819863807175
# 10828.jpg:      0.5394210473621704
# 1402.jpg:       0.7717948069526598
# 11187.jpg:      0.8990542320343551
# 11269.jpg:      1.0152568105342692
# 10005.jpg:      1.726865458942667
# 11606.jpg:      2.503699315835493
# 11150.jpg:      10.481921282304823
# 10309.jpg:      1.7976931348623157e+308
# 10456.jpg:      1.7976931348623157e+308

# Контурите се зачувани во папката result
# Споредбата на лист со дршка со лист без дршка дава многу мала сличност и обратно (функцијата matchShapes враќа голем број за тие што се малку слични а мал за тие што се многу слични).
# Резултатите се печатат по редослед на сличност, во растечки редослед (според добиената бројка се почнува од најмногу сличните).