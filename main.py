
import torch

from algorithms.policy import ConvActorPolicy
from algorithms.reinforce import REINFORCE
from envs.minesweeper import DummyVecEnv, MineSweeper


def test(policy):
    env = MineSweeper((6, 6), 6, 30)
    for j in range(10):
        print('-'*150)
        obs = env.reset()
        for i in range(10):
            env.render()
            obs = torch.tensor(obs, dtype=torch.float32)
            action, logits = policy.act(obs.unsqueeze(0))
            print(f'action = {action}')
            print(f'logits: {logits}')
            obs, reward, done, info = env.step(action.item())
            print(f'reward: {reward}, done: {done}')
            if done:
                env.render()
                break


env_args = {
    'board_shape': (6, 6),
    'num_mines': 6,
    'max_steps': 30,
}
size = env_args['board_shape'][0] * env_args['board_shape'][1]
vec_env = DummyVecEnv(lambda args: MineSweeper(**args),
                      n=32, env_args=env_args)
# policy = SimpleActorPolicy([size, 64, 64, size], optim_lr=1e-2)
policy = ConvActorPolicy(32, 16, [1, 2, 3, 5], optim_lr=3e-4, one_hotify=True)
print(
    f'Number of parameters: {sum(p.numel() for p in policy.parameters() if p.requires_grad)}')
algo = REINFORCE(policy, vec_env, gamma=0, baseline='none') #gamma=0.1
# test(policy)
policy.train()
algo.train(10_000)
policy.eval()
test(policy)
