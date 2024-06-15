import os
import glob
import cv2
import numpy as np

def load_images(directory: str) -> list[np.ndarray]:
    image_files = glob.glob(os.path.join(directory, '*'))
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in image_files]
    return images

def compute_sift_features(image: np.ndarray) -> tuple:
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_descriptors(desc1, desc2) -> list:
    bf_matcher = cv2.BFMatcher()
    raw_matches = bf_matcher.knnMatch(desc1, desc2, k=2)
    good_matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
    return good_matches

def main() -> None:
    database_path = 'database/'
    images_path = 'images/'
    
    search_image_name = input('Enter the name of the search image: ')
    search_image_path = os.path.join(images_path, search_image_name)

    poster_images = load_images(database_path)
    poster_features = [compute_sift_features(image) for image in poster_images]

    print(f"Loading image from: {search_image_path}")

    search_image = cv2.imread(search_image_path, cv2.IMREAD_GRAYSCALE)

    if search_image is None:
        print(f"Error: Unable to load image from {search_image_path}. Please check the file path and try again.")
        return

    search_keypoints, search_descriptors = compute_sift_features(search_image)

    matches = [match_descriptors(desc[1], search_descriptors) for desc in poster_features]
    best_match_index = max(range(len(matches)), key=lambda i: len(matches[i]))

    best_poster = poster_images[best_match_index]
    best_poster_keypoints, _ = poster_features[best_match_index]

    search_image_with_keypoints = cv2.drawKeypoints(search_image, search_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    best_poster_with_keypoints = cv2.drawKeypoints(best_poster, best_poster_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    concatenated_images = np.concatenate((search_image, best_poster), axis=1)
    concatenated_keypoints = np.concatenate((search_image_with_keypoints, best_poster_with_keypoints), axis=1)

    cv2.imshow('Original Images', concatenated_images)
    cv2.imshow('Keypoints', concatenated_keypoints)
    cv2.imshow('Matches', cv2.drawMatchesKnn(search_image, search_keypoints, best_poster, best_poster_keypoints, matches[best_match_index], None, flags=2))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()