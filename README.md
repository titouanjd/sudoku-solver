# Sudoku solver (WIP)
A 9x9 Sudoku puzzle is detected from an image using a CNN trained on the MNIST dataset.
A recursive backtracking algorithm then solves the puzzle.

## Project current state
- Sudoku solver working from a valid grid
- Current issue: the trained CNN model sometimes makes wrong digit predictions leading to invalid grids which the Sudoku solver cannot solve