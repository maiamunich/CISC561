import random
import sys
from read import readInput
from write import writeOutput

from host import GO

class MyPlayer3():
    def __init__(self):
        self.type = 'random'

    def get_input(self, go, piece_type):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''        
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check = True):
                    possible_placements.append((i,j))

        if not possible_placements:
            return "PASS"
        else:
            return random.choice(possible_placements)

    def capture_number_pieces(self, go, piece_type):
        '''
        Calculate the number of pieces that can be captured.
        '''
        capture_number = 0
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check = True):
                    capture_number += 1
        return capture_number

    def liberty_number_pieces(self, go, piece_type):
        '''
        Calculate the number of pieces that have liberty.
        '''
        liberty_number = 0
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check = True):
                    liberty_number += 1
        return liberty_number

    def avoid_capture_pieces(self, go, piece_type):
        '''
        Calculate the number of pieces that can be captured.
        '''
        avoid_capture_number = 0
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check = True):
                    avoid_capture_number += 1
        return avoid_capture_number

    def avoid_liberty_pieces(self, go, piece_type):
        '''
        Calculate the number of pieces that have liberty.
        '''
        avoid_liberty_number = 0
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check = True):
                    avoid_liberty_number += 1
        return avoid_liberty_number

    def heuristic(self, go, piece_type):
        capture_number = self.capture_number_pieces(go, piece_type)
        liberty_number = self.liberty_number_pieces(go, piece_type)
        avoid_capture_number = self.avoid_capture_pieces(go, piece_type)
        avoid_liberty_number = self.avoid_liberty_pieces(go, piece_type)
        return capture_number - liberty_number + avoid_capture_number - avoid_liberty_number

    def get_best_move(self, go, piece_type):
        '''
        Get the best move for the player.
        '''
        best_move = None
        best_score = -float('inf')
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check = True):
                    go.board[i][j] = piece_type
                    score = self.heuristic(go, piece_type)
                    go.board[i][j] = 0
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
        return best_move
if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = MyPlayer3()
    action = player.get_input(go, piece_type)
    writeOutput(action)