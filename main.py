from sudoku import Sudoku
import warnings
import math
import numpy as np
import cv2 as cv
import skimage as ski
import keras


def show_img(img):
    cv.imshow(f'Sudoku grid {img.shape}', img)
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

    return img, gray, thresh

def find_sudoku_contour(thresh):
    """Detect Sudoku grid contours."""
    contours, _ = cv.findContours(thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    # based on point coordinates, find largest 4-point area - most likely to be full sudoku grid
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    for cont in contours:
        epsilon = 0.04 * cv.arcLength(cont, closed=True)  # 4% of contour's perimeter
        polygon = cv.approxPolyDP(cont, epsilon=epsilon, closed=True)  # approximate contour
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

    # source points
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    # side length of square
    side = int(max(
        math.dist(top_left, top_right),
        math.dist(top_right, bottom_right),
        math.dist(bottom_right, bottom_left),
        math.dist(bottom_left, top_left)
    ))

    # destination points
    dst = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype='float32')

    # compute perspective transform matrix and apply it to image
    matrix = cv.getPerspectiveTransform(src, dst)
    warped_img = cv.warpPerspective(img, matrix, dsize=(side, side))

    return warped_img

def extract_digit(cell):
    _, thresh = cv.threshold(cell, thresh=127, maxval=255, type=cv.THRESH_BINARY_INV)
    thresh = ski.segmentation.clear_border(thresh)  # remove borders
    contours, _ = cv.findContours(thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

    # if no contours were found than this is an empty cell
    if len(contours) == 0:
        return

    # otherwise, find the largest contour in the cell and create a mask for it
    largest_contour = max(contours, key=cv.contourArea)
    mask = np.zeros(thresh.shape, dtype='uint8')
    cv.drawContours(mask, contours=[largest_contour], contourIdx=-1, color=255, thickness=-1)

    # compute the percentage of masked pixels relative to the total image area
    (h, w) = thresh.shape
    percent_filled = cv.countNonZero(mask) / (w * h)

    # if less than 3% of the mask is filled then it is noise, can ignore
    if percent_filled < 0.03:
        return

    # apply the mask to the thresholded cell
    digit = cv.bitwise_and(thresh, thresh, mask=mask)
    return digit

def extract_digits(warped_img, model):
    """Extract digits from Sudoku grid using CNN model."""

    cell_size = warped_img.shape[0] // 9
    digits = []

    for r in range(9):
        row_digits = []
        for c in range(9):
            x, y = c * cell_size, r * cell_size
            cell = warped_img[y:y + cell_size, x:x + cell_size]
            digit = extract_digit(cell)
            if digit is not None:
                # preprocess cell for digit recognition
                sized = cv.resize(digit, dsize=(28, 28))  # resize to 28x28 as CNN expects that

                # normalise to [0, 1] and reshape for cnn model (height, width, channels)
                cell_input = sized.astype("float32") / 255
                cell_input = cell_input.reshape(1, 28, 28, 1)

                # predict the digit using the CNN model
                pred = np.argmax(model.predict(cell_input))
                row_digits.append(pred)

            else:
                row_digits.append(0)
        digits.append(row_digits)

    return digits

def main(img_path: str = 'puzzles/sudoku.jpg'):
    img, gray, thresh = preprocess_image(img_path)
    polygon = find_sudoku_contour(thresh)

    if polygon is None:
        warnings.warn('Could not find Sudoku grid.')
        return

    warped_img = get_warped_image(gray, polygon)

    # load pre-trained CNN
    model = keras.models.load_model('cnn_mnist_model.keras')

    grid = extract_digits(warped_img, model)

    sudoku = Sudoku(grid)
    if not sudoku.valid:
        print(sudoku)
        return

    print(sudoku)
    sudoku.solve()
    print(sudoku)


if __name__ == "__main__":
    main()
