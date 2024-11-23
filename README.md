# Sudoku solver
A 9x9 Sudoku puzzle is detected from an image using a CNN trained on the MNIST dataset. A recursive backtracking algorithm then solves the puzzle.

## Procedure
1. Load a picture of a Sudoku puzzle
2. Detect the Sudoku grid
3. For each cell on the grid:
    1. Extract the digit from the cell
    2. Use a pre-trained CNN to predict the digit value
4. Solve the detected Sudoku using a backtracking algorithm

## Example use

### Original sudoku puzzle
Load a picture of a Sudoku puzzle.

<img src="images/original.jpg" width="400">

### Grayscale image
Convert the colored image to a single grayscale channel for alignment with MNIST digits.

<img src="images/gray.jpg" width="400">

### Binary image
Convert the image to black and white to better detect the grid borders.

<img src="images/thresh.jpg" width="400">

### Warped image
Warp the sudoku grid to fit the frame.

<img src="images/warp.jpg" width="400">

### Detected grid
Extract the Sudoku grid from the image by looping over each cell and predicting its value using the pre-trained CNN.

<img src="images/grid.jpg" width="300">

### Solution
Solve the Sudoku puzzle using a backtracking algorithm.

<img src="images/solution.jpg" width="300">