import gym
import torch
import torch.nn.functional as F

from algorithms.policy import Policy


# TODO: add support for recurrent policies
class REINFORCE():
    def __init__(self, policy: Policy, env: gym.Env, gamma: float, baseline: str = 'batch_norm') -> None:
        '''
        params:
            policy: a policy that can act given an observation and return actions and logits
            env: a vectorized environment that returns an np array as observations batch along with rewards and other stuff
            gamma: discount factor for future rewards
            baseline: Either 'batch_norm' or 'none'. Determines wheter a baseline will be used to normalize Q values.

        I assume that env returns a numpy batch of observations of shape N x obs.shape
        '''
        super().__init__()
        self.policy = policy
        self.optimizer = policy.optimizer
        self.env = env
        self.gamma = gamma
        self.baseline = baseline

    def train(self, train_iters: int):
        for i in range(train_iters):
            obs_buffer, act_buffer, logits_buffer, rew_buffer, done_buffer, info_buffer = [
            ], [], [], [], [], []
            obs_buffer.append(torch.tensor(
                self.env.reset(), dtype=torch.float32))
            '''
            obs_buffer[t]: the batch of observations at time-step t
            At time t, the policy uses obs[t] to generate act[t], logits[t].
            Using obs[t], act[t], the environment generates obs[t+1], rew[t], done[t]
            '''
            finished = False
            # roll out the current policy to obtain trajectories
            # I don't have to roll out for one full episode: I can roll out each env for a fixed number of steps and reset them if need be. But then computing Q values would be a bit different.
            while not finished:
                actions, logits = self.policy.act(obs_buffer[-1])
                obs, reward, done, info = self.env.step(
                    actions.cpu().flatten().numpy())
                obs_buffer.append(torch.tensor(obs, dtype=torch.float32))
                act_buffer.append(actions)
                logits_buffer.append(logits)
                rew_buffer.append(torch.tensor(reward, dtype=torch.float32))
                done_buffer.append(torch.tensor(done))
                info_buffer.append(info)
                finished = torch.all(done_buffer[-1] == True)

            # compute the returns for these policies
            # Q_esitmates[t] is the MC estimate of Q(s_t, a_t)
            Q_estimates = [0]
            for reward in rew_buffer[::-1]:
                Q_estimates.append(self.gamma * Q_estimates[-1] + reward)
            Q_estimates = Q_estimates[1:]
            Q_estimates = Q_estimates[::-1]

            # compute the loss
            # T x B is zero for all indices where done = True and 1 for all those where done = False
            active_mask = ~torch.stack(done_buffer)
            Q_estimates = torch.stack(Q_estimates).to(
                self.policy.device)  # TxB
            mean_reward = Q_estimates[0].mean()
            if self.baseline == 'batch_norm':
                Q_estimates[active_mask] -= Q_estimates[active_mask].mean()
                # TODO ?
                Q_estimates[active_mask] /= (
                    Q_estimates[active_mask].std() + 1e6)
            act_buffer = torch.stack(act_buffer)  # T x B
            logits_buffer = torch.stack(logits_buffer)  # T x B x num_actions
            neg_log_likelihoods = F.cross_entropy(logits_buffer.flatten(
                0, 1), act_buffer.flatten(0, 1), reduction='none').reshape(act_buffer.shape)  # T x B
            loss = (neg_log_likelihoods*active_mask*Q_estimates)[active_mask].sum() / \
                torch.count_nonzero(active_mask)

            # update the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

            # TODO: logging
            if i % 50 == 49:
                print(
                    f'Iteration {i+1}: loss = {loss:.4}, avg episode value = {mean_reward:.4}, average q value: {Q_estimates[active_mask].mean():.4}, avg episode length = {torch.count_nonzero(active_mask)/active_mask.shape[1]:.4}')

    @torch.no_grad()
    def eval(self, ):
        pass
