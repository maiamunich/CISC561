import json
import random
from pathlib import Path

from read import readInput
from write import writeOutput

from host import GO

class MyPlayer3():
    CAPTURE_WEIGHT = 10
    LIBERTY_WEIGHT = 1
    SAFETY_WEIGHT = 5
    MAX_SEARCH_DEPTH = 2
    MAX_BRANCHING_FACTOR = 10
    Q_LEARNING_RATE = 0.2
    Q_DISCOUNT = 0.9
    Q_EXPLORATION = 0.15
    TOURNAMENT_STRATEGIES = ('random', 'greedy', 'aggressive', 'alpha_beta', 'q_learning')
    TOURNAMENT_REPEATS = 3
    TOURNAMENT_VERSION = 1

    # Initializes tournament selection and the default strategy.
    def __init__(self):
        self.type = 'tournament'
        self.move_number = 0

    # Routes each move to the tournament winner or selected algorithm.
    def get_input(self, go, piece_type):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''   
        self.move_number += 1
        if self.type == 'tournament':
            self.type = self.get_tournament_winner(go.size)
        if self.type == 'random':
            return self.get_random_move(go, piece_type)
        if self.type == 'greedy':
            return self.get_best_move(go, piece_type)
        if self.type == 'aggressive':
            return self.get_aggressive_move(go, piece_type)
        if self.type == 'alpha_beta':
            return self.get_alpha_beta_move(go, piece_type)
        if self.type == 'q_learning':
            return self.get_q_learning_move(go, piece_type)
        return self.get_best_move(go, piece_type)

    # Tournament selector: compares every strategy and caches the winner.
    def get_tournament_winner(self, board_size):
        result_path = Path(__file__).resolve().with_name('my_player3_tournament.json')
        try:
            with result_path.open('r') as result_file:
                saved_results = json.load(result_file)
            if (
                saved_results.get('version') == self.TOURNAMENT_VERSION
                and saved_results.get('board_size') == board_size
                and saved_results.get('winner') in self.TOURNAMENT_STRATEGIES
            ):
                return saved_results['winner']
        except (OSError, ValueError, AttributeError):
            pass

        standings = {
            strategy: {'wins': 0, 'draws': 0, 'losses': 0, 'points': 0, 'margin': 0}
            for strategy in self.TOURNAMENT_STRATEGIES
        }
        match_number = 0
        for first_index, first_strategy in enumerate(self.TOURNAMENT_STRATEGIES):
            for second_strategy in self.TOURNAMENT_STRATEGIES[first_index + 1:]:
                for repeat in range(self.TOURNAMENT_REPEATS):
                    for first_is_black in (True, False):
                        random.seed(match_number + repeat * 1000)
                        match_number += 1
                        first_player = MyPlayer3()
                        first_player.type = first_strategy
                        second_player = MyPlayer3()
                        second_player.type = second_strategy
                        game = GO(board_size)

                        if first_is_black:
                            black_player, white_player = first_player, second_player
                            first_piece = 1
                        else:
                            black_player, white_player = second_player, first_player
                            first_piece = 2

                        winner = game.play(black_player, white_player)
                        second_piece = 3 - first_piece
                        first_score = game.score(first_piece) + (game.komi if first_piece == 2 else 0)
                        second_score = game.score(second_piece) + (game.komi if second_piece == 2 else 0)
                        margin = first_score - second_score
                        standings[first_strategy]['margin'] += margin
                        standings[second_strategy]['margin'] -= margin

                        if winner == 0:
                            standings[first_strategy]['draws'] += 1
                            standings[second_strategy]['draws'] += 1
                            standings[first_strategy]['points'] += 1
                            standings[second_strategy]['points'] += 1
                        elif winner == first_piece:
                            standings[first_strategy]['wins'] += 1
                            standings[first_strategy]['points'] += 3
                            standings[second_strategy]['losses'] += 1
                        else:
                            standings[second_strategy]['wins'] += 1
                            standings[second_strategy]['points'] += 3
                            standings[first_strategy]['losses'] += 1

        winner = max(
            self.TOURNAMENT_STRATEGIES,
            key=lambda strategy: (
                standings[strategy]['points'],
                standings[strategy]['wins'],
                standings[strategy]['margin'],
            ),
        )
        results = {
            'version': self.TOURNAMENT_VERSION,
            'board_size': board_size,
            'winner': winner,
            'standings': standings,
        }
        try:
            with result_path.open('w') as result_file:
                json.dump(results, result_file)
        except OSError:
            pass
        return winner

    # Q-learning: load and save values across the host's per-turn processes.
    def load_q_data(self):
        q_path = Path(__file__).resolve().with_name('my_player3_qvalues.json')
        try:
            with q_path.open('r') as q_file:
                data = json.load(q_file)
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict) or not isinstance(data.get('values'), dict):
            data = {}
        data.setdefault('values', {})
        pending = data.get('pending')
        if isinstance(pending, dict) and 'state' in pending:
            color = pending.get('piece_type')
            data['pending'] = {str(color): pending} if color in (1, 2) else {}
        elif not isinstance(pending, dict):
            data['pending'] = {}
        return q_path, data

    # Q-learning: persist the table and pending transition.
    def save_q_data(self, q_path, data):
        try:
            with q_path.open('w') as q_file:
                json.dump(data, q_file)
        except OSError:
            pass

    # Q-learning: identify a state, including its ko-relevant previous board.
    def q_state_key(self, go, piece_type):
        board_key = '/'.join(''.join(str(cell) for cell in row) for row in go.board)
        previous_key = '/'.join(''.join(str(cell) for cell in row) for row in go.previous_board)
        return str(piece_type) + ':' + board_key + ':' + previous_key

    # Q-learning: reward uses change in this player's stone margin.
    def score_margin(self, go, piece_type):
        return go.score(piece_type) - go.score(3 - piece_type)

    # Q-learning: enumerate actions allowed in the current state.
    def legal_moves(self, go, piece_type):
        moves = []
        for row in range(go.size):
            for column in range(go.size):
                if go.valid_place_check(row, column, piece_type, test_check=True):
                    moves.append((row, column))
        return moves

    # Q-learning: update the prior state-action value from the observed reward.
    def update_q_values(self, data, go, piece_type, state_key, legal_moves):
        pending_by_color = data.get('pending')
        if not isinstance(pending_by_color, dict):
            data['pending'] = {}
            return
        pending = pending_by_color.pop(str(piece_type), None)
        if not isinstance(pending, dict):
            return

        if pending.get('piece_type') != piece_type:
            return
        if not any(cell for row in go.board for cell in row):
            return

        values = data['values']
        previous_state = pending.get('state')
        previous_action = pending.get('action')
        if not isinstance(previous_state, str) or not isinstance(previous_action, str):
            return

        previous_values = values.setdefault(previous_state, {})
        old_value = previous_values.get(previous_action, 0.0)
        next_values = values.get(state_key, {})
        next_actions = [f'{row},{column}' for row, column in legal_moves]
        best_next_value = max(
            (next_values.get(action, 0.0) for action in next_actions),
            default=0.0,
        )
        reward = self.score_margin(go, piece_type) - pending.get('score_margin', 0)
        previous_values[previous_action] = old_value + self.Q_LEARNING_RATE * (
            reward + self.Q_DISCOUNT * best_next_value - old_value
        )

    # Random baseline: choose uniformly from legal moves.
    def get_random_move(self, go, piece_type):
        possible_moves = []
        for row in range(go.size):
            for column in range(go.size):
                if go.valid_place_check(row, column, piece_type, test_check=True):
                    possible_moves.append((row, column))
        return random.choice(possible_moves) if possible_moves else "PASS"

    # Q-learning strategy: explore or choose the highest-valued legal action.
    def get_q_learning_move(self, go, piece_type):
        q_path, data = self.load_q_data()
        moves = self.legal_moves(go, piece_type)
        state_key = self.q_state_key(go, piece_type)
        self.update_q_values(data, go, piece_type, state_key, moves)

        if not moves:
            self.save_q_data(q_path, data)
            return 'PASS'

        state_values = data['values'].get(state_key, {})
        if random.random() < self.Q_EXPLORATION:
            move = random.choice(moves)
        else:
            best_value = max(
                (state_values.get(f'{row},{column}', 0.0) for row, column in moves),
                default=0.0,
            )
            best_moves = [
                move for move in moves
                if state_values.get(f'{move[0]},{move[1]}', 0.0) == best_value
            ]
            move = random.choice(best_moves)

        data['pending'][str(piece_type)] = {
            'state': state_key,
            'action': f'{move[0]},{move[1]}',
            'piece_type': piece_type,
            'score_margin': self.score_margin(go, piece_type),
        }
        self.save_q_data(q_path, data)
        return move

    # Shared board features used by greedy, aggressive, and alpha-beta play.
    def count_group_liberties(self, go, row, column):
        liberties = set()
        for member_row, member_column in go.ally_dfs(row, column):
            for neighbor_row, neighbor_column in go.detect_neighbor(member_row, member_column):
                if go.board[neighbor_row][neighbor_column] == 0:
                    liberties.add((neighbor_row, neighbor_column))
        return len(liberties)

    # Shared board feature: count stones in groups with only one liberty.
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

    # Greedy strategy: score a single move using captures, liberties, and safety.
    def heuristic(self, go, piece_type, move, captured_number, at_risk_before):
        liberty_number = self.count_group_liberties(go, move[0], move[1])
        at_risk_after = self.count_at_risk_pieces(go, piece_type)
        avoided_capture_number = at_risk_before - at_risk_after
        return (
            self.CAPTURE_WEIGHT * captured_number
            + self.LIBERTY_WEIGHT * liberty_number
            + self.SAFETY_WEIGHT * avoided_capture_number
        )

    # Shared by aggressive and alpha-beta search to simulate legal candidates.
    def simulate_move(self, go, move, piece_type):
        trial_go = go.copy_board()
        trial_go.previous_board = [row[:] for row in go.board]
        trial_go.board[move[0]][move[1]] = piece_type
        trial_go.update_board(trial_go.board)
        captured_pieces = trial_go.remove_died_pieces(3 - piece_type)
        trial_go.died_pieces = captured_pieces
        return trial_go, len(captured_pieces)

    # Alpha-beta: evaluate a board from the root player's perspective.
    def evaluate_state(self, go, piece_type):
        opponent_type = 3 - piece_type
        komi_adjustment = go.komi if piece_type == 2 else -go.komi
        stone_margin = go.score(piece_type) - go.score(opponent_type) + komi_adjustment
        own_at_risk = self.count_at_risk_pieces(go, piece_type)
        opponent_at_risk = self.count_at_risk_pieces(go, opponent_type)
        return 100 * stone_margin + 5 * (opponent_at_risk - own_at_risk)

    # Alpha-beta: order and cap legal moves to the configured branch factor.
    def get_ordered_moves(self, go, piece_type, root_type, maximizing):
        moves = []
        for row in range(go.size):
            for column in range(go.size):
                move = (row, column)
                if not go.valid_place_check(row, column, piece_type, test_check=True):
                    continue
                next_go, _ = self.simulate_move(go, move, piece_type)
                moves.append((self.evaluate_state(next_go, root_type), move, next_go))

        moves.sort(key=lambda candidate: candidate[0], reverse=maximizing)
        return [
            (move, next_go)
            for _, move, next_go in moves[:self.MAX_BRANCHING_FACTOR]
        ]

    # Alpha-beta: recursive minimax search with pruning.
    def alpha_beta_search(self, go, depth, root_type, current_type, alpha, beta):
        if depth == 0:
            return self.evaluate_state(go, root_type)

        maximizing = current_type == root_type
        moves = self.get_ordered_moves(go, current_type, root_type, maximizing)
        if not moves:
            return self.alpha_beta_search(
                go,
                depth - 1,
                root_type,
                3 - current_type,
                alpha,
                beta,
            )

        if maximizing:
            best_score = -float('inf')
            for _, next_go in moves:
                score = self.alpha_beta_search(
                    next_go,
                    depth - 1,
                    root_type,
                    3 - current_type,
                    alpha,
                    beta,
                )
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score

        best_score = float('inf')
        for _, next_go in moves:
            score = self.alpha_beta_search(
                next_go,
                depth - 1,
                root_type,
                3 - current_type,
                alpha,
                beta,
            )
            best_score = min(best_score, score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score

    # Alpha-beta strategy: return the best root move from the bounded search.
    def get_alpha_beta_move(self, go, piece_type):
        moves = self.get_ordered_moves(go, piece_type, piece_type, maximizing=True)
        if not moves:
            return "PASS"

        best_move = None
        best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        for move, next_go in moves:
            score = self.alpha_beta_search(
                next_go,
                self.MAX_SEARCH_DEPTH - 1,
                piece_type,
                3 - piece_type,
                alpha,
                beta,
            )
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_move

    # Aggressive strategy: find the largest immediate capture available.
    def get_max_capture_count(self, go, piece_type):
        max_capture_count = 0
        for row in range(go.size):
            for column in range(go.size):
                if not go.valid_place_check(row, column, piece_type, test_check=True):
                    continue
                _, captured_count = self.simulate_move(go, (row, column), piece_type)
                max_capture_count = max(max_capture_count, captured_count)
        return max_capture_count

    # Aggressive strategy: maximize captures while accounting for capture risk.
    def get_aggressive_move(self, go, piece_type):
        best_move = None
        best_score = -float('inf')
        at_risk_before = self.count_at_risk_pieces(go, piece_type)

        for row in range(go.size):
            for column in range(go.size):
                move = (row, column)
                if not go.valid_place_check(row, column, piece_type, test_check=True):
                    continue

                trial_go, captured_number = self.simulate_move(go, move, piece_type)
                opponent_type = 3 - piece_type
                opponent_best_capture = self.get_max_capture_count(trial_go, opponent_type)

                liberty_number = self.count_group_liberties(trial_go, row, column)
                at_risk_after = self.count_at_risk_pieces(trial_go, piece_type)
                avoided_capture_number = at_risk_before - at_risk_after
                score = (
                    self.CAPTURE_WEIGHT * (captured_number - opponent_best_capture)
                    + self.LIBERTY_WEIGHT * liberty_number
                    + self.SAFETY_WEIGHT * avoided_capture_number
                )
                if score > best_score:
                    best_score = score
                    best_move = move

        return best_move if best_move is not None else "PASS"

    # Greedy strategy: choose the legal move with the highest one-ply heuristic.
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
    input_path = Path("input.txt")
    if not input_path.is_file():
        input_path = Path(__file__).resolve().with_name("input.txt")
    piece_type, previous_board, board = readInput(N, input_path)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = MyPlayer3()
    action = player.get_input(go, piece_type)
    writeOutput(action)