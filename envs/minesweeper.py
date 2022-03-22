from enum import Enum
from typing import Any, Dict, Tuple

import gym
import numpy as np


class Status(Enum):
    UNK = -1
    MINE = -2
    FLAG = -3
    ZERO, I, II, III, IV, V, VI, VII, VIII, IX = range(10)


class MineSweeper(gym.Env):
    def __init__(self, board_shape: Tuple[int, int], num_mines: int) -> None:
        super().__init__()
        self.BAD_ACTION_REWARD = -1
        self.GOOD_ACTION_REWARD = 1
        self.WIN_REWARD = 100
        self.LOSE_REWARD = -100
        self.board_shape = board_shape
        self.num_mines = num_mines
        self.ground_truth = None
        self.obs_state = None
        self.mines = []
        self.done = False

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            # no changes
            return self.obs_state.copy(), 0, True, None
        if not self.is_valid(action[0], action[1]):
            # illegal action, small negative rewrad
            return self.obs_state.copy(), self.BAD_ACTION_REWARD, self.done, None
        if self.obs_state[action] != Status.UNK.value:
            # redundant action, small negative reward
            return self.obs_state.copy(), self.BAD_ACTION_REWARD, self.done, None
        if self.ground_truth[action] == Status.MINE.value:
            # game over, lost, large negative reward
            self.obs_state[action] = Status.MINE.value
            self.done = True
            return self.obs_state.copy(), self.LOSE_REWARD, self.done, None

        reward = self.GOOD_ACTION_REWARD
        # modifies self.obs_state to unlock unknown cells
        self.unlock_cell(action)
        if np.count_nonzero(self.obs_state == Status.UNK.value) == self.num_mines:
            # game is won, all remaining unk cells are mines
            reward += self.WIN_REWARD
            self.done = True
        return self.obs_state.copy(), reward, self.done, None

    def reset(self) -> Any:
        self.ground_truth, self.mines = self.generate_board()
        self.obs_state = Status.UNK.value * np.ones_like(self.ground_truth)
        return self.obs_state

    def render(self, mode: str = 'console') -> None:
        if mode == 'console':
            for i in range(self.board_shape[0]):
                row = ''
                for j in range(self.board_shape[1]):
                    temp = int(self.obs_state[i, j])
                    temp = '-' if temp == Status.UNK.value else temp
                    temp = '*' if temp == Status.MINE.value else temp
                    row += str(temp) + '    '
                print(row)

    def seed(self, seed: int = 23):
        np.random.seed(seed)

    def generate_board(self):
        board = np.zeros(self.board_shape)
        mines = np.random.choice(
            np.prod(self.board_shape), self.num_mines, replace=False)
        mine_coordinates = []
        h, w = self.board_shape
        for mine in mines:
            x, y = int(mine / w), int(mine) % w
            mine_coordinates.append((x, y))
            for nx in [x-1, x, x+1]:
                for ny in [y-1, y, y+1]:
                    if self.is_valid(nx, ny) and board[nx, ny] != Status.MINE.value:
                        board[nx, ny] += 1
            board[x, y] = Status.MINE.value
        return board, mine_coordinates

    def unlock_cell(self, coordinates):
        x, y = coordinates
        if self.obs_state[x, y] != Status.UNK.value or self.ground_truth[x, y] == Status.MINE.value:
            # is already unlocked or is a mine
            return
        self.obs_state[x, y] = self.ground_truth[x, y]
        if self.ground_truth[x, y] == 0:
            # recursively unlock neighbouring cells
            for nx in [x-1, x, x+1]:
                for ny in [y-1, y, y+1]:
                    if self.is_valid(nx, ny):
                        self.unlock_cell((nx, ny))

    def is_valid(self, x: int, y: int):
        return 0 <= x < self.board_shape[0] and 0 <= y < self.board_shape[1]


# env = MineSweeper((6, 6), 6)
# env.reset()
# print(f'mines: {env.mines}')
# env.render()
# for i in range(10):
#     x = int(input())
#     y = int(input())
#     obs, rew, done, info = env.step((x, y))
#     print(f'obs.shape: {obs.shape}, reward: {rew}, done: {done}')
#     env.render()