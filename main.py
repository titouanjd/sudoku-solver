from sudoku import Sudoku
import math
import numpy as np
import cv2 as cv
import keras


def show_img(img):
    cv.imshow('Sudoku grid', img)
    cv.waitKey()
    cv.destroyAllWindows()

def preprocess_image(img_path):
    """Preprocess the Sudoku image"""
    img = cv.imread(img_path)

    # convert to grayscale
    gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)

    # gaussian blur for noise reduction
    blurred = cv.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    # black and white
    _, thresh = cv.threshold(blurred, thresh=127, maxval=255, type=cv.THRESH_BINARY_INV)

    return img, thresh

def find_sudoku_contour(thresh):
    """Detect grid contours."""
    contours, _ = cv.findContours(thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)  # return external contours (mode) as list of points, only the end points of straight lines are stored (method)
    largest_contour = max(contours, key=cv.contourArea)  # based on point coordinates, find largest area - most likely to be full sudoku grid
    epsilon = 0.04 * cv.arcLength(largest_contour, closed=True)  # 4% of largest contour's perimeter
    polygon = cv.approxPolyDP(largest_contour, epsilon=epsilon, closed=True)
    if len(polygon) == 4:  # It's the Sudoku grid
        return polygon
    return None

# Warp the perspective to get the top-down view of the grid
def get_warped_image(img, polygon):
    """Store Sudoku grid in square image"""
    # identify the four corners of the polygon
    # bottom-right point has the largest (x + y) value
    # top-left point has the smallest (x + y) value
    # top-right point has largest (x - y) value
    # bottom-left point has smallest (x - y) value
    sums = [pt[0][0] + pt[0][1] for pt in polygon]
    diffs = [pt[0][0] - pt[0][1] for pt in polygon]
    bottom_right = polygon[max(range(len(polygon)), key=lambda i: sums[i])][0]
    top_left = polygon[min(range(len(polygon)), key=lambda i: sums[i])][0]
    top_right = polygon[max(range(len(polygon)), key=lambda i: diffs[i])][0]
    bottom_left = polygon[min(range(len(polygon)), key=lambda i: diffs[i])][0]

    # source po
    source_points = np.array([top_left, top_right, bottom_right, bottom_left],
                             dtype='float32')
    
    # side length of square
    side = int(max(
        math.dist(top_left, top_right),
        math.dist(top_right, bottom_right),
        math.dist(bottom_right, bottom_left),
        math.dist(bottom_left, top_left)
    ))

    dest_points = np.array([[0, 0], [side, 0], [side, side], [0, side]],
                           dtype='float32')
    
    # compute perspective transform matrix and apply it to image
    matrix = cv.getPerspectiveTransform(src=source_points, dst=dest_points)
    warped_img = cv.warpPerspective(img, matrix, dsize=(side, side))

    return warped_img

def extract_digits(warped_img, model):
    pass

def main(img_path: str = 'puzzles/sudoku_1.jpg'):
    img, thresh = preprocess_image(img_path)
    polygon = find_sudoku_contour(thresh)

    if polygon is None:
        print("Couldn't find Sudoku grid.")
        return
    
    warped_img = get_warped_image(img, polygon)
    print(img.shape)
    print(warped_img.shape)

    # load pre-trained CNN
    model = keras.models.load_model('cnn_mnist.keras')

    digits = extract_digits(warped_img, model)


if __name__ == "__main__":
    main()
