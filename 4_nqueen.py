

class nqueen():

    def __init__(self,n):
        self.n = n 
        self.board = [[0 for _ in range(n)] for __ in range(n)]
        self.solutions = []

    def solve(self):
        self.backtrack(0)
        return self.solutions
    
    def backtrack(self, col):
        if col == self.n:
            solution = []
            for row in range(self.n):
                solution.append(tuple(self.board[row]))
            self.solutions.append(solution)
            return
        
        for row in range(self.n):
            if self.is_valid(row,col):
                self.board[row][col] = 1
                self.backtrack(col + 1)
                self.board[row][col] = 0

    def is_valid(self, row, col):
        for i in range(col):
            if self.board[row][i] == 1:
                return False
        
        i, j = row, col
        while i>= 0 and j >= 0:

            if self.board[i][j] == 1:
                return False
            i -= 1
            j -= 1

        i, j = row, col
        while i < self.n and j >= 0:
            if self.board[i][j] == 1:
                return False
            i += 1
            j -= 1
        
        return True
    
n = 6
queens = nqueen(n)
solutions = queens.solve()

print(f"Number of solutions for {n}-queens problem: {len(solutions)}")

for i, solution in enumerate(solutions):
    print(f'Solution {i+1}:')
    for row in solution:
        print(row)
    print()

