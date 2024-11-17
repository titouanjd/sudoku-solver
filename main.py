from sudoku import Sudoku
import warnings
import math
import numpy as np
import cv2 as cv
import skimage as ski
import keras


def show_img(img: np.ndarray) -> None:
    """Show an image on screen."""
    cv.imshow(f'Sudoku grid {img.shape}', img)
    cv.waitKey()
    cv.destroyAllWindows()

def preprocess_image(img_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess the Sudoku image."""
    img = cv.imread(img_path)

    # convert to grayscale
    gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)

    # gaussian blur for noise reduction
    blurred = cv.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    # black and white
    _, thresh = cv.threshold(blurred, thresh=127, maxval=255, type=cv.THRESH_BINARY_INV)

    return img, gray, thresh

def get_grid_contour(thresh: np.ndarray) -> np.ndarray | None:
    """Detect Sudoku grid contours."""
    contours, _ = cv.findContours(thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    # based on point coordinates, find largest 4-point area - most likely to be full sudoku grid
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    for cont in contours:
        epsilon = 0.04 * cv.arcLength(cont, closed=True)  # 4% of contour's perimeter
        polygon = cv.approxPolyDP(cont, epsilon=epsilon, closed=True)  # approximate contour
        if len(polygon) == 4:  # it's the Sudoku grid
            return polygon
    warnings.warn('Could not find Sudoku grid.')
    return None

def warp_image(img: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Warp Sudoku grid to store it in a square image."""
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

def extract_grid(warped_img: np.ndarray) -> list[list[int]]:
    """
    Extract the Sudoku grid digits from a warped image. The value of each
    extracted digit is predicted using a pre-trained CNN model.
    """
    # load the CNN model
    model = keras.models.load_model('cnn_mnist_model.keras')

    # loop over grid cells to extract the digits and make predictions
    cell_size = warped_img.shape[0] // 9
    grid = []
    for r in range(9):
        grid_row = []
        for c in range(9):
            x, y = c * cell_size, r * cell_size
            cell = warped_img[y:y + cell_size, x:x + cell_size]
            digit = extract_digit(cell)
            if digit is not None:
                # resize to 28x28 for the CNN model
                sized = cv.resize(digit, dsize=(28, 28))

                # normalise to [0, 1] and reshape for the CNN model
                digit_input = sized.astype('float32') / 255
                digit_input = digit_input.reshape(1, 28, 28, 1)

                # predict the digit using the CNN model
                pred = np.argmax(model.predict(digit_input, verbose=0))
                grid_row.append(pred)

            else:
                grid_row.append(0)
        grid.append(grid_row)

    return grid

def extract_digit(cell: np.ndarray) -> np.ndarray | None:
    """
    Extract the digit from a Sudoku a cell.
    Scale the digit to 75% of the cell height.
    Center the digit on the cell.
    """
    _, thresh = cv.threshold(cell, thresh=127, maxval=255, type=cv.THRESH_BINARY_INV)
    thresh = ski.segmentation.clear_border(thresh)  # remove borders
    contours, _ = cv.findContours(thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

    # if no contours were found then this is an empty cell
    if not contours:
        return None

    # otherwise, find the bounding box of the largest contour in the cell
    largest_contour = max(contours, key=cv.contourArea)
    x, y, width, height = cv.boundingRect(largest_contour)

    # if less than 3% of the cell is filled then it is noise, can ignore
    if width * height < 0.03 * cell.size:
        return None

    # crop the digit from the cell
    cropped_digit = thresh[y:y + height, x:x + width]

    # scale the digit to 75% of the cell height
    new_height = int(cell.shape[0] * 0.75)
    scale_ratio = new_height / height
    new_width = int(width * scale_ratio)
    cropped_digit = cv.resize(cropped_digit, dsize=(new_width, new_height))

    # center the digit in the cell
    digit = np.zeros_like(cell)
    x_offset = (digit.shape[0] - new_width) // 2
    y_offset = (digit.shape[1] - new_height) // 2
    digit[y_offset:y_offset + new_height,
          x_offset:x_offset + new_width] = cropped_digit

    return digit

def main(img_path: str = 'puzzles/sudoku_1.jpg'):
    img, gray, thresh = preprocess_image(img_path)

    polygon = get_grid_contour(thresh)

    if polygon is None:
        return

    warped_img = warp_image(gray, polygon)

    grid = extract_grid(warped_img)

    sudoku = Sudoku(grid)
    print(sudoku)

    if not sudoku.valid:
        return

    sudoku.solve()
    print(sudoku)


if __name__ == "__main__":
    main()
