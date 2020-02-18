import numpy as np
from termcolor import colored
from numba import njit
import time
import pandas as pd


def generate_start_state(num_cols:int):
    col_pos = np.repeat(1, num_cols)
    forward_steps = num_cols
    while forward_steps > 0:
        col = np.random.choice(np.arange(num_cols), 1)[0]
        if col_pos[col] < 3:
            col_pos[col] += 1
            forward_steps -= 1
    return col_pos


def print_board(board_state):
    for row in reversed(board_state):
        string = ''
        for num in row:
            if num == 0:
                string += colored('X ', 'red')
            elif num == 1:
                string += colored('X ', 'blue')
            else:
                string += colored('0 ', 'white')
        print(string)
    print('')


@njit
def flip_board_state(board_state):
    fbs = board_state[::-1]
    fbs_copy = fbs.copy()
    for i in range(board_state.shape[0]):
        for j in range(board_state.shape[1]):
            if fbs[i, j] == 0:
                fbs_copy[i, j] = 1
            elif fbs[i, j] == 1:
                fbs_copy[i, j] = 0
    return fbs_copy


@njit
def calc_row_sum(board_state):
    h, w = board_state.shape
    row_sum = []
    for i in range(h):
        this_row = 0
        for j in range(w):
            if (board_state[i, j] == 0) or (board_state[i, j] == 1):
                this_row += 1
        row_sum.append(this_row)
    return row_sum


@njit
def check_available_actions(board_state):
    # action format: list with allowed actions. Each action is a 2 value list: [column, step_size]
    # step_size is the number of steps forward (positive) or backward on that column.
    available_actions = []
    rowsum = calc_row_sum(board_state)
    for col in range(board_state.shape[1]):
        pos0 = np.where(board_state[:, col] == 0)[0][0]
        pos1 = np.where(board_state[:, col] == 1)[0][0]
        step_size = rowsum[pos0]
        if (pos0 + step_size < board_state.shape[0]) & (pos0 + step_size != pos1):
            available_actions.append([col, step_size])
        if (pos0 - step_size >= 0) & (pos0 - step_size != pos1):
            available_actions.append([col, -step_size])
    return available_actions


@njit
def update_board_state(board_state, col, step):
    available_actions = check_available_actions(board_state)
    new_board_state = board_state.copy()
    if [col, step] in available_actions:
        cur_row = np.where(board_state[:, col] == 0)[0]
        new_board_state[cur_row, col] = -1
        new_board_state[cur_row + step, col] = 0
    else:
        print('Action not permitted!')
        new_board_state = None
    return new_board_state


@njit
def calc_state_value(board_state, attack_factor=1.0):
    h, w = board_state.shape
    value = 0.0
    for i in range(h):
        for j in range(w):
            if board_state[i, j] == 0:
                value += i * attack_factor
            elif board_state[i, j] == 1:
                value -= (h - 1 - i)
    return value


# Calculate the value of a board_state.
# value is the position of player_0 minus position of player_1.
# Penalty is used the penalize a player putting >= 2 on the second last row, >=3 on the third last row, etc.
@njit
def value_function(board_state, penalty):
    h, w = board_state.shape
    value = 0.0
    row_sum = calc_row_sum(board_state=board_state)
    for i in range(h):
        if (row_sum[i] > (h - 1 - i)) and i < (h - 1):
            value -= penalty
        for j in range(w):
            if board_state[i, j] == 0:
                value += i
            elif board_state[i, j] == 1:
                value -= (h - 1 - i)
    return value


@njit
def check_win_conditions(board_state):
    winning_player = -1
    if (board_state[-1, :] == 0).sum() == board_state.shape[1]:
        winning_player = 0
    elif (board_state[0, :] == 1).sum() == board_state.shape[1]:
        winning_player = 1
    return winning_player


# Search tree algorithm. Recursive function that looks ahead 'depth' turns, and determines the best path forward, assuming ideal opponent behavior.
def minimax_treesearch(board_state, depth: int, penalty, printing: bool = False):
    if depth == 0:
        state_value = value_function(board_state=board_state, penalty=penalty)
        winning_player = check_win_conditions(board_state)
        if winning_player == 0:
            state_value = np.inf
        elif winning_player == 1:
            state_value = -np.inf
        return state_value, None
    else:
        values = []
        available_actions = check_available_actions(board_state)
        for a in available_actions:
            new_board_state = update_board_state(board_state, col=a[0], step=a[1])
            state_value, action = minimax_treesearch(board_state=flip_board_state(new_board_state), depth=depth - 1, penalty=penalty, printing=printing)
            winning_player = check_win_conditions(new_board_state)
            state_value *= -1
            if winning_player == 0:
                state_value = np.inf
            elif winning_player == 1:
                state_value = -np.inf
            values.append(state_value)
        max_value = max(values)
        max_indices = []
        for val, i in zip(values, range(len(values))):
            if val >= max_value:
                max_indices.append(i)
        rand_index = max_indices[np.random.randint(0, len(max_indices))]
        best_action = available_actions[rand_index]
        if printing:
            added_tabs = '\t' * depth
            print(added_tabs + "Actions: ", available_actions)
            print(added_tabs + "Values:  ", values)
            print(added_tabs + "Best action: %s, Max value: %s" % (best_action, max_value))
        return max_value, best_action


# Used to create a Ligato game board
class LigatoGame:
    def __init__(self, board_size):
        self.player_names = ['Red', 'Blue']
        self.board_size = board_size
        self.board_state = None
        self.turn = 0
        self.cur_player = 0
        self.game_finished = False
        self.winner = None
        self.player_AI = [None, None]
        self.printing = True

        # initialize random start board_state
        self.random_state(start=True)

    def set_player_AI(self, player, depth, penalty):
        ai = LigatoAI(depth=depth, penalty=penalty)
        self.player_AI[player] = ai
        if self.printing:
            print("Player %s AI %s initiated." % (player, ai.name))

    def print(self):
        print_board(self.board_state)

    def random_state(self, start=True, seed=12345):
        self.turn = 0
        self.cur_player = 0
        self.game_finished = False
        board = np.ones(shape=self.board_size, dtype='int8') * -1
        np.random.seed(seed)
        if start:
            size = int(self.board_size[1])
            # Player 0 start position
            start_player0 = generate_start_state(num_cols=size)
            # Player 1 start position
            start_player1 = generate_start_state(num_cols=size)
            for col in range(self.board_size[1]):
                board[start_player0[col], col] = 0
                board[self.board_size[0] - 1 - start_player1[col], col] = 1
        else:
            for col in range(self.board_size[1]):
                positions = np.random.choice(range(self.board_size[0]), 2, replace=False)
                board[positions[0], col] = 0
                board[positions[1], col] = 1
        self.board_state = board
        if self.printing:
            self.print()

    def get_board_state(self, player):
        if player == 0:
            board_state = self.board_state.copy()
        else:
            board_state = flip_board_state(self.board_state.copy())
        return board_state

    def set_board_state(self, player, board_state):
        if player == 0:
            self.board_state = board_state
        else:
            self.board_state = flip_board_state(board_state)

    def _move(self, player, action):
        if player != self.cur_player:
            print("Wrong player! This turn is for player %s" % self.cur_player)
            return
        if self.game_finished:
            print("Game is finished. Won by player %s." % self.check_win_conditions())
            return
        board_state = self.get_board_state(player=player)
        new_board_state = update_board_state(board_state=board_state, col=action[0], step=action[1])
        if new_board_state is None:
            return
        self.turn += 1
        self.cur_player = (self.cur_player + 1) % 2
        self.set_board_state(player=player, board_state=new_board_state)
        if self.printing:
            self.print()
        self.check_win_conditions()

    def play(self):
        pl = self.cur_player
        if self.printing:
            print("Turn ", self.turn)
        if self.player_AI[pl] is None:
            print("Human player %s is up for play:" % pl)
            col = input('Which column [0-' + str(self.board_size[1] - 1) + '] ?')
            step = input('Stepsize? ')
            self._move(player=pl, action=[int(col), int(step)])
        else:
            start_time = time.time()
            action = self.player_AI[pl].play(board_state=self.get_board_state(player=pl))
            if self.printing:
                print("AI %s is playing:" % pl)
                print("AI picks action %s in %s seconds." % (action, time.time() - start_time))
            self._move(player=pl, action=action)
        if not self.game_finished:
            self.play()

    def check_win_conditions(self):
        winning_player = check_win_conditions(self.board_state)
        if winning_player != -1:
            if self.printing:
                print("Player %s won in %s turns!" % (winning_player, self.turn))
            self.game_finished = True
            self.winner = winning_player
        if self.turn > 30 * self.board_size[1]:
            self.game_finished = True
            self.winner = -1
            if self.printing:
                print("After %s turns, the game was stopped with no winner." % (30 * self.board_size[1]))


# Basic Ligato AI, using minimax treesearch algorithm to determine its moves.
# WIP: gives an error sometimes when depth is > 4.
class LigatoAI:
    def __init__(self, depth, penalty):
        self.depth = depth
        self.name = "Minimax treesearch"
        self.penalty = penalty
        # self.folder = os.path.join(os.path.curdir, name)
        # if not os.path.exists(self.folder):
        #    os.makedirs(self.folder)

    def play(self, board_state):
        _, action = minimax_treesearch(board_state=board_state, depth=self.depth, penalty=self.penalty, printing=False)
        return action


# Class to create a tournament with AI's, letting them compete, gather statistics and save them.
class LigatoTournament:
    def __init__(self, name):
        self.name = name
        self.log = pd.DataFrame(columns=['board_h', 'board_w', 'p0_depth', 'p0_penalty', 'p1_depth', 'p1_penalty',
                                         'won_by', 'turns'])

    def write_log(self):
        self.log.to_csv(self.name + '.csv')

    def battle(self, board_size, num_games, d_range, p_range):
        game = LigatoGame(board_size)
        game.printing = False
        for d0 in range(*d_range):
            for d1 in range(*d_range):
                for p0 in range(*p_range):
                    for p1 in range(*p_range):
                        game.set_player_AI(0, d0, p0)
                        game.set_player_AI(1, d1, p1)
                        print("AI 0: d %s p %s. AI 1: d %s p %s" % (d0, p0, d1, p1))
                        winner = []
                        for n in range(num_games):
                            game.random_state(start=True, seed=n)
                            game.play()
                            print("Winner: %s" % game.winner)
                            self.log = self.log.append({'board_h': board_size[0],
                                             'board_w': board_size[1],
                                             'p0_depth': int(d0),
                                             'p0_penalty': p0,
                                             'p1_depth': d1,
                                             'p1_penalty': p1,
                                             'won_by': game.winner,
                                             'turns': game.turn}, ignore_index=True)
        self.write_log()
        self.log.groupby(['p0_depth', 'p1_depth', 'won_by']).turns.count()
