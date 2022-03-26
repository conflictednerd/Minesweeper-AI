from typing import Any, Dict, List, Tuple, Union

import gym
import numpy as np


class Status():
    UNK = -1
    MINE = -2
    FLAG = -3
    ZERO, I, II, III, IV, V, VI, VII, VIII, IX = range(10)


class MineSweeper(gym.Env):
    def __init__(self, board_shape: Tuple[int, int], num_mines: int, max_steps: int) -> None:
        super().__init__()
        self.BAD_ACTION_REWARD = -1
        self.GOOD_ACTION_REWARD = 1
        self.WIN_REWARD = 100
        self.LOSE_REWARD = -1
        self.board_shape = board_shape
        self.num_mines = num_mines
        self.max_steps = max_steps
        self.current_step = 0
        self.ground_truth = None
        self.obs_state = None
        self.mines = []
        self.done = False

    def step(self, action: Union[Tuple[int, int], int]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if not isinstance(action, tuple):
            action = (int(action / self.board_shape[1]), int(action) % self.board_shape[1])
        self.current_step += 1
        if self.current_step > self.max_steps:
            self.done = True
        if self.done:
            # no changes
            return self.obs_state.copy(), 0, True, None
        if not self.is_valid(action[0], action[1]):
            # illegal action, small negative rewrad
            return self.obs_state.copy(), self.BAD_ACTION_REWARD, self.done, None
        if self.obs_state[action] != Status.UNK:
            # redundant action, small negative reward
            return self.obs_state.copy(), self.BAD_ACTION_REWARD, self.done, None
        if self.ground_truth[action] == Status.MINE:
            # game over, lost, large negative reward
            self.obs_state[action] = Status.MINE
            self.done = True
            return self.obs_state.copy(), self.LOSE_REWARD, self.done, None

        reward = self.GOOD_ACTION_REWARD
        # modifies self.obs_state to unlock unknown cells
        self.unlock_cell(action)
        if np.count_nonzero(self.obs_state == Status.UNK) == self.num_mines:
            # game is won, all remaining unk cells are mines
            reward += self.WIN_REWARD
            self.done = True
        return self.obs_state.copy(), reward, self.done, None

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.done = False
        self.ground_truth, self.mines = self.generate_board()
        self.obs_state = Status.UNK * np.ones_like(self.ground_truth)
        return self.obs_state

    def render(self, mode: str = 'console') -> None:
        if mode == 'console':
            for i in range(self.board_shape[0]):
                row = ''
                for j in range(self.board_shape[1]):
                    temp = int(self.obs_state[i, j])
                    temp = '-' if temp == Status.UNK else temp
                    temp = '*' if temp == Status.MINE else temp
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
                    if self.is_valid(nx, ny) and board[nx, ny] != Status.MINE:
                        board[nx, ny] += 1
            board[x, y] = Status.MINE
        return board, mine_coordinates

    def unlock_cell(self, coordinates):
        x, y = coordinates
        if self.obs_state[x, y] != Status.UNK or self.ground_truth[x, y] == Status.MINE:
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


class DummyVecEnv(gym.Env):
    def __init__(self, env_fn, n, env_args: Dict = {}) -> None:
        self.envs = [env_fn(env_args) for _ in range(n)]

    def reset(self) -> List[np.ndarray]:
        return np.array([env.reset() for env in self.envs])

    def step(self, actions) -> Tuple[np.ndarray, List[float], List[bool], List[Dict[str, Any]]]:
        obss, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(action)
            obss.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return np.array(obss), np.array(rewards), np.array(dones), infos

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


# env_args = {
#     'board_shape': (6, 6),
#     'num_mines': 10,
#     'max_steps': 40,
# }
# vec_env = DummyVecEnv(lambda args: MineSweeper(**args),
#                       n=32, env_args=env_args)
# obs = vec_env.reset()
# print(len(obs))
# print(obs[0].shape)
# for i in range(20):
#     obs, rew, done, info = vec_env.step([(i, i)]*32)
#     print(len(obs))
#     print(rew)
#     print(done)
