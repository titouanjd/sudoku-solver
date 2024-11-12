class Sudoku:
    def __init__(self, grid: list[list[int]]) -> None:
        self.grid = grid

        # Pre-define row, column, and box sets for quick constraint checks
        self.rows = [set() for _ in range(9)]
        self.cols = [set() for _ in range(9)]
        self.boxes = [[set() for _ in range(3)] for _ in range(3)]

        # Validate the grid
        if not self.is_valid_grid():
            print(self)
            raise Exception('Sudoku grid is invalid.')

    def is_valid_grid(self) -> bool:
        """Check if Sudoku grid is valid."""
        for r in range(9):
            for c in range(9):
                num = self.grid[r][c]
                if num == 0:  # Ignore empty cells
                    continue
                
                # Check row
                if num in self.rows[r]:
                    return False
                self.rows[r].add(num)

                # Check column
                if num in self.cols[c]:
                    return False
                self.cols[c].add(num)

                # Check 3x3 box
                box_row, box_col = r // 3, c // 3
                if num in self.boxes[box_row][box_col]:
                    return False
                self.boxes[box_row][box_col].add(num)

        return True

    def is_valid(self, r: int, c: int, k: int) -> bool:
        """Check if placing k at (r, c) is valid according to Sudoku rules."""
        if k in self.rows[r] or k in self.cols[c] or k in self.boxes[r//3][c//3]:
            return False
        return True

    def solve(self, r: int = 0, c: int = 0) -> bool:
        """Solve Sudoku puzzle."""
        if r == 9:  # if end of grid reached, final solution found
            return True
        elif c == 9:  # if last column reached, change row
            return self.solve(r+1, 0)
        elif self.grid[r][c] != 0:  # if cell already filled, change column
            return self.solve(r, c+1)
        else:
            for k in range(1, 10):  # try all k=0-9 values
                if self.is_valid(r, c, k):  # check if k is valid at (r, c)
                    self.grid[r][c] = k
                    self.rows[r].add(k)
                    self.cols[c].add(k)
                    self.boxes[r//3][c//3].add(k)
                    if self.solve(r, c+1):
                        return True
                    self.grid[r][c] = 0  # reset cell value if impossible
                    self.rows[r].remove(k)
                    self.cols[c].remove(k)
                    self.boxes[r//3][c//3].remove(k)
            return False

    def __str__(self) -> str:
        """Return a string representation of the Sudoku grid."""
        grid_str = ""
        for i in range(9):
            if i % 3 == 0 and i != 0:
                grid_str += "----- + ----- + -----\n"

            for j in range(9):
                if j % 3 == 0 and j != 0:
                    grid_str += "| "

                grid_str += str(self.grid[i][j]) if self.grid[i][j] != 0 else "."
                
                if j != 9 - 1:
                    grid_str += " "
                else:
                    grid_str += "\n"

        return grid_str

def main():
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

    # create sudoku puzzle from grid
    sudoku = Sudoku(grid)
    print(sudoku)

    # solve puzzle
    sudoku.solve()
    print(sudoku)


if __name__ == "__main__":
    main()
