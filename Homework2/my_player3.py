from read import readInput
from write import writeOutput

from host import GO

class MyPlayer3():
    CAPTURE_WEIGHT = 10
    LIBERTY_WEIGHT = 1
    SAFETY_WEIGHT = 5

    def __init__(self):
        self.type = 'random'

    def get_input(self, go, piece_type):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''        
        return self.get_best_move(go, piece_type)

    def count_group_liberties(self, go, row, column):
        liberties = set()
        for member_row, member_column in go.ally_dfs(row, column):
            for neighbor_row, neighbor_column in go.detect_neighbor(member_row, member_column):
                if go.board[neighbor_row][neighbor_column] == 0:
                    liberties.add((neighbor_row, neighbor_column))
        return len(liberties)

    def count_at_risk_pieces(self, go, piece_type):
        visited = set()
        at_risk_number = 0
        for row in range(go.size):
            for column in range(go.size):
                if go.board[row][column] != piece_type or (row, column) in visited:
                    continue
                group = set(go.ally_dfs(row, column))
                visited.update(group)
                if self.count_group_liberties(go, row, column) == 1:
                    at_risk_number += len(group)
        return at_risk_number

    def heuristic(self, go, piece_type, move, captured_number, at_risk_before):
        liberty_number = self.count_group_liberties(go, move[0], move[1])
        at_risk_after = self.count_at_risk_pieces(go, piece_type)
        avoided_capture_number = at_risk_before - at_risk_after
        return (
            self.CAPTURE_WEIGHT * captured_number
            + self.LIBERTY_WEIGHT * liberty_number
            + self.SAFETY_WEIGHT * avoided_capture_number
        )

    def get_best_move(self, go, piece_type):
        '''
        Get the best move for the player.
        '''
        best_move = None
        best_score = -float('inf')
        at_risk_before = self.count_at_risk_pieces(go, piece_type)
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check = True):
                    trial_go = go.copy_board()
                    trial_go.board[i][j] = piece_type
                    trial_go.update_board(trial_go.board)
                    captured_pieces = trial_go.remove_died_pieces(3 - piece_type)
                    score = self.heuristic(
                        trial_go,
                        piece_type,
                        (i, j),
                        len(captured_pieces),
                        at_risk_before,
                    )
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
        return best_move if best_move is not None else "PASS"
if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = MyPlayer3()
    action = player.get_input(go, piece_type)
    writeOutput(action)