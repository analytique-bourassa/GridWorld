We have that chapter 3 (p.66, 69 and 70)
$$
P_{ss^\prime}^a = P(s_{t+1} = s^{\prime}|s_{t} = s, a_t = a)
$$

$$
R_{ss^\prime}^a = E(r_{t+1}|s_{t} = s, a_t = a, s_{t+1} = s^\prime)
$$

$$
V^\pi(s) = E_\pi[R_t|s_t = s] = E_\pi [\sum_{k=0}^\infty \gamma^k r_{t + k + 1}| s_t = s ]
$$

$V^\pi$ : state-value function for policy $\pi$ 

$ Q^\pi(s,a) $: action-value function for policy $\pi$
$$
Q^\pi(s,a) = E_\pi [\sum_{k=0}^\infty \gamma^k r_{t + k + 1}| s_t = s, a_t = a]
$$
Bellman equation:
$$
V^\pi(s) = \sum_{a}\pi(s,a)\sum_{s^\prime}P_{ss^\prime}^a[R_{ss^\prime}^a + \gamma V^\pi(s^\prime)]
$$
NB: reader used to see the equations is Typora 