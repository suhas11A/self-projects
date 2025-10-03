import logging

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging.INFO)

class Sudoku:
    def __init__(self, board):
        self.board = board

    def is_valid(self, row, col, num):
        # Check row
        if any(self.board[row][j] == num for j in range(9)):
            return False
        # Check column
        if any(self.board[i][col] == num for i in range(9)):
            return False
        # Check 3x3 box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if self.board[start_row + i][start_col + j] == num:
                    return False
        return True

    def find_empty(self):
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    return i, j
        return None

    def solve(self):
        empty = self.find_empty()
        if not empty:
            return True  # Solved
        row, col = empty
        for num in range(1, 10):
            if self.is_valid(row, col, num):
                self.board[row][col] = num
                if self.solve():
                    return True
                self.board[row][col] = 0  # backtrack
        return False

    def display(self):
        for i, row in enumerate(self.board):
            if i % 3 == 0 and i != 0:
                print("-" * 21)
            for j, val in enumerate(row):
                if j % 3 == 0 and j != 0:
                    print("|", end=" ")
                print(val if val != 0 else ".", end=" ")
            print()

if __name__ == "__main__":
    # Example puzzle (0 = empty)
    puzzle = [
        [5, 1, 7, 6, 0, 0, 0, 3, 4],
        [2, 8, 9, 0, 0, 4, 0, 0, 0],
        [3, 4, 6, 2, 0, 5, 0, 9, 0],
        [6, 0, 2, 0, 0, 0, 0, 1, 0],
        [0, 3, 8, 0, 0, 6, 0, 4, 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 9, 0, 0, 0, 0, 0, 7, 8],
        [7, 0, 3, 4, 0, 0, 5, 6, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    sudoku = Sudoku(puzzle)
    logging.info("Initial board:")
    sudoku.display()
    if sudoku.solve():
        logging.info("Solved board:")
        sudoku.display()
    else:
        logging.warning("No solution exists.")

