# Sudoku solver
A 9x9 Sudoku puzzle is detected from an image using a CNN trained on the MNIST dataset.
A recursive backtracking algorithm then solves the puzzle.

## Procedure
1. Convert Sudoku image to black and white
2. Detect Sudoku grid and warp the image
3. For each cell on the grid
    1. Extract digit from the cell
    2. Use pre-trained CNN to make prediction about digit
4. Solve Sudoku using backtracking algorithm