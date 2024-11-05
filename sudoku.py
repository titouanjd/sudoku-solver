class Sudoku:
    def __init__(self, grid: list[list]) -> None:
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])

    def is_valid(self, r: int, c: int, k: int) -> bool:
        """Check if placing k at (r, c) is valid according to Sudoku rules."""
        # Check row
        if k in self.grid[r]:
            return False

        # Check column
        if k in [self.grid[i][c] for i in range(self.width)]:
            return False

        # Check box
        if k in [self.grid[i][j] for i in range(r//3*3, r//3*3+3) for j in range(c//3*3, c//3*3+3)]:
            return False

        return True

    def solve(self, r: int = 0, c: int = 0) -> bool:
        """Solve Sudoku puzzle."""
        if r == self.height:  # if end of grid reached, final solution found
            return True
        elif c == self.width:  # if last column reached, change row
            return self.solve(r+1, 0)
        elif self.grid[r][c] != 0:  # if cell already filled, change column
            return self.solve(r, c+1)
        else:
            for k in range(1, 10):  # try all k=0-9 values
                if self.is_valid(r, c, k):  # check if k is valid at (r, c)
                    self.grid[r][c] = k
                    if self.solve(r, c+1):
                        return True
                    self.grid[r][c] = 0  # reset cell value if impossible
            return False

    def __str__(self) -> str:
        """Return a string representation of the Sudoku grid."""
        grid_str = ""
        for i in range(self.height):
            if i % 3 == 0 and i != 0:
                grid_str += "----- + ----- + -----\n"

            for j in range(self.width):
                if j % 3 == 0 and j != 0:
                    grid_str += "| "

                grid_str += str(self.grid[i][j]) if self.grid[i][j] != 0 else "."
                
                if j != self.width - 1:
                    grid_str += " "
                else:
                    grid_str += "\n"

        return grid_str.strip()  # Remove trailing newline


if __name__ == "__main__":
    grid = [
        [0, 0, 4, 0, 5, 0, 0, 0, 0],
        [9, 0, 0, 7, 3, 4, 6, 0, 0],
        [0, 0, 3, 0, 2, 1, 0, 4, 9],
        [0, 3, 5, 0, 9, 0, 4, 8, 0],
        [0, 9, 0, 0, 0, 0, 0, 3, 0],
        [0, 7, 6, 0, 1, 0, 9, 2, 0],
        [3, 1, 0, 9, 7, 0, 2, 0, 0],
        [0, 0, 9, 1, 8, 2, 0, 0, 3],
        [0, 0, 0, 0, 6, 0, 1, 0, 0]
    ]
    sudoku = Sudoku(grid)
    print(sudoku)
    sudoku.solve()
    print()
    print(sudoku)
