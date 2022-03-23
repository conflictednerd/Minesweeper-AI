# REINFORCE

Also known as Monte Carlo policy gradient. It is a gradient based approach to solving the RL problem. Vanilla REINFORCE is an on-policy algorithm, but with some slight modification we can also use it in off-policy settings (remember <font color='red'>off-policy $\rightarrow$ behaviour policy $\neq$ target policy</font>).
$$
\newcommand{\E}{\mathbb {E}}
\newcommand{\P}{\mathbb {P}}
$$


## Notation

| Symbol                     | Meaning                                            |
| -------------------------- | -------------------------------------------------- |
| $s \in \mathcal{S}$, $S_t$ | state, random variable of state at time step $t$   |
| $a \in \mathcal A$, $A_t$  | action, random variable of action at time step $t$ |
| $r \in \mathcal R$, $R_t$  | reward, random variable of reward at time step $t$ |
|                            |                                                    |
|                            |                                                    |
|                            |                                                    |
|                            |                                                    |



## Derivation

Let $\pi_\theta(a|s)$ be a stochastic policy parametrized by $\theta$. The objective of an RL algorithm is to maximize the expected discounted future reward, that is
$$
\max_\theta \E_{\tau\sim p_\theta(\tau)} \left[ \sum_{t=0}^{\infty} {\gamma^t r(s_t, a_t)} \right] = \max_\theta J(\theta).
$$


Policy gradient methods are built upon the idea that we can maximize this objective by using its gradient with respect to $\theta$, and performing a gradient ascent to find the best value for the parameters $\theta$. So basically, we would like to find $\grad_\theta J(\theta)$. All of the stuff that follows are basically us trying to compute this gradient efficiently.

Let's start by expanding the expected value:
$$
J(\theta) = \E_{\tau \sim p_\theta(\tau)}[r(\tau)] = \sum_{\tau}p_\theta(\tau)r(\tau)
$$
We are summing over all possible trajectories, and computing the product of the probability of that trajectory with its total reward.

Now, let's compute the gradient:
$$
\grad_\theta J(\theta) = \sum_{\tau} {\grad_\theta p_\theta(\tau)r(\tau)}
$$
Notice that $r_\theta(\tau)$ does not depend on $\theta$ and we can treat it like a constant. By multiplying and dividing by $p_\theta(\tau)$ and using the log-derivative trick, we can write:
$$
\grad_\theta J(\theta) = \sum_{\tau} {p_\theta(\tau) \frac{\grad_\theta p_\theta(\tau)}{p_\theta(\tau)}r(\tau)} = \sum_{\tau} {p_\theta(\tau) r(\tau)\grad_\theta \log {p_\theta(\tau)}}
$$
Why did we rewrote the gradient so that it has a logarithmic form? Because $p_\theta (\tau)$ is the product of a number of terms, some of which are not dependent on $\theta$. By taking its logarithm, we can eliminate these terms when computing the gradient. Observe the following:
$$
\log p_\theta (\tau) = \log {\left(\P[S_0=s_0]\prod_{t=0}\pi_\theta(a_t|s_t)\P[S_{t+1}=s_{t+1}|S_t = s_t, A_t=a_t]\right)} \\ 
= \log \P[S_0 = s_0] + \sum_{t=0} {\P[S_{t+1} = s_{t+1}|S_t = s_t, A_{t} = a_t]} + \sum_{t=0} {\log\pi_\theta(a_t|s_t)}
$$
See how $\theta$ appears only in the last sum, where the log probabilities of actions are summed. Using this, we could write the gradient as:
$$
\grad_\theta J(\theta) = \sum_\tau {p_\theta(\tau)\left(r(\tau)\sum_t {\grad_\theta \log\pi_\theta(a_t|s_t)}\right)}
$$
Writing this as an expectation (over the trajectory distribution) we have
$$
\grad_\theta J(\theta) = \E_{\tau \sim p_\theta(\tau)} \left[\left(\sum_t \grad_\theta \log \pi_\theta(a_t|s_t)\right)\left(\sum_t\gamma^tr(s_t, a_t)\right)\right]
$$
Finally, we can obtain a Monte Carlo estimate of $\grad_\theta J(\theta)$ by replacing the expectation in the above formula with the mean of empirical samples.

Once we have this gradient, updating the parameters is easy:
$$
\theta \leftarrow \theta + \alpha\grad_\theta J(\theta)
$$
There are two important tricks that we can use to reduce the variance of this estimator.

**Causality Trick:** Think about what we are doing by using aforementioned gradient. For a trajectory, we are modifying $\theta$ such that the (log) probability of each action in that trajectory is increased or decreased based on how rewarding this trajectory has been. In other words, if the total discounted sum of rewards for a trajectory is a large positive number, by multiplying it with the grad log terms, we are increasing the probability of selecting all of the actions that we have encountered in that trajectory. Similarly, we are reducing the probability of actions in trajectories where the total discounted reward is negative. **But**, this means that we are using all obtained rewards, including those that were obtained prior to selecting action $a_t$ to determine how good $a_t$ has been. This is not ideal, because how good selecting $a_t$ has been, is independent of the previous rewards. Knowing this, we can modify the policy gradient so that an action is not penalized or rewarded for what has happened in the past:
$$
\grad_\theta J(\theta) = \E_{\tau\sim p_\theta(\tau)}\left[\sum_t \grad_\theta\log\pi_\theta(a_t|s_t)\sum_{t'=t}\gamma^{t'-t}r(s_{t'}, a_{t'})\right]
$$
The innermost sum is the _reward-to-go_ from time $t$. Essentially, it is a Monte Carlo estimate of $Q(s_t, a_t)$ that we have obtained by sampling the trajectory $\tau$. We can denote it by $\hat Q(s_t, a_t)$ and write
$$
\grad_\theta J(\theta) = \E_{\tau \sim p_\theta (\tau)} \left[\sum_t \hat{Q}(s_t, a_t) \grad_\theta \log \pi_\theta(a_t|s_t)\right].
$$
[*Using this notation, you may start to think about using a more elegant estimate of $Q(s_t, a_t)$ to reduce the variance of gradient estimator. You are correct! We don't have to use a full Monte Carlo simulation to get our estimate of $Q(s,a)$, we can instead learn the $Q$ function and bootstrap our estimate using it, similar to temporal difference methods. This is the basis for actor-critic methods.*]

**Baselines:** As was mentioned in the previous discussion, $\hat{Q}(s_t, a_t)$ determines if selecting $a_t$ when we were at $s_t$ was a good decision or not. Now imagine a scenario where we only get positive rewards. Then all of the actions that we select are _good_, in the sense that $\hat{Q}(s_t, a_t)$ is positive. Thus any update is increasing the probability of selecting all of the actions in the sampled trajectory. You can see that this will slow down the convergence to the optimal policy specially when the action space is large. This is where the idea of baselines come to play: if we normalize $\hat{Q}(s_t, a_t)$ values such that in a trajectory it is sometimes positive and sometimes negative, then we can converge to a policy that selects the best actions more quickly. We can easily prove that adding a constant value $b$ to the sum of rewards of a trajectory will keep the policy gradient unbiased. A very convenient choice of baseline is to use the average observed reward, which is not optimal but works pretty good!
$$
b = \E_{\tau\sim p_\theta (\tau)}{[r(\tau)]} \\
\grad_\theta J(\theta) = \E_{\tau \sim p_\theta (\tau)} \left[\sum_t (\hat{Q}(s_t, a_t) - b) \grad_\theta \log \pi_\theta(a_t|s_t)\right]
$$


## Algorithm

1. Sample $\{\tau^i\}_{i=1}^{N}$ trajectories by running $\pi_\theta(a|s)$
2. [Optionally] compute the baseline $b = \frac{1}{N}\sum\limits_{i=1}^N{r(\tau)}$ or just normalize all $\hat{Q}$ values
3. Estimate the gradient using $\grad_\theta J(\theta) \approx \frac{1}{N}\sum\limits_{i=1}^{N}{\sum\limits_{t}(\hat{Q}(s_t, a_t)-b)\grad_\theta\log\pi_\theta(a^i_t|s^i_t)}$
4. Update the parameters $\theta \leftarrow \theta + \alpha \grad_\theta J(\theta)$

## Final Remarks

The REINFORCE algorithm as was presented here is an on-policy algorithm: we need to sample trajectories $\tau^i$ using the current policy $\pi_\theta(a|s)$ to compute the gradient. However, with some minor changes and ideas from importance sampling, we can make it work with trajectories that are sampled from a different [behaviour] policies. The basic idea is to use the following identity
$$
\E_{x\sim p(x)} [f(x)] = \E_{x\sim q(x)}[\frac{p(x)}{q(x)}f(x)].
$$
This allows us to define $J(\theta)$ when trajectories are sampled from a different policy. We can compute the gradient of $J(\theta)$ in this new form, however we need to be cautious as the $p(x)$ term in the above identity will be the probability of trajectory under the distribution induced by $\pi_\theta$ and is therefore dependent on $\theta$. Thus, we need to account for it in the gradient computation. I will not get into further details here.

