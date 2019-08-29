'''
## Prioritised Experience Replay (PER) Memory ##
# Adapted from: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# Creates prioritised replay memory buffer to add experiences to and sample batches of experiences from
'''

import numpy as np
import random
import redis

from utils.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create replay buffer.
        Args:
            size (int): max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, gamma):
        data = (obs_t, action, reward, obs_tp1, done, gamma)

        self._storage.append(data)

        self._next_idx += 1

    def remove(self, num_samples):
        del self._storage[:num_samples]
        self._next_idx = len(self._storage)

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, gammas = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, gamma = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            gammas.append(gamma)
        return [np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(
            gammas)]

    def sample(self, batch_size, **kwags):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        gammas: np.array
            product of gammas for N-step returns
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        weights = np.zeros(len(idxes))
        inds = np.zeros(len(idxes))
        return self._encode_sample(idxes) + [weights, inds]


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        self.it_capacity = 1
        while self.it_capacity < size * 2:  # We use double the soft capacity of the PER for the segment trees to allow for any overflow over the soft capacity limit before samples are removed
            self.it_capacity *= 2

        self._it_sum = SumSegmentTree(self.it_capacity)
        self._it_min = MinSegmentTree(self.it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        assert idx < self.it_capacity, "Number of samples in replay memory exceeds capacity of segment trees. Please increase capacity of segment trees or increase the frequency at which samples are removed from the replay memory"

        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def remove(self, num_samples):
        super().remove(num_samples)
        self._it_sum.remove_items(num_samples)
        self._it_min.remove_items(num_samples)

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        gammas: np.array
            product of gammas for N-step returns
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class RedisReplayBuffer(object):
    def __init__(self, config):
        self.state_inds = (0, config['state_dims'])
        self.action_inds = (config['state_dims'], config['state_dims'] + config['action_dims'])
        self.reward_inds = (self.action_inds[1], self.action_inds[1] + 1)
        self.next_state_inds = (self.reward_inds[1], self.reward_inds[1] + config['state_dims'])
        self.dones_inds = (self.next_state_inds[1], self.next_state_inds[1] + 1)
        self.gammas_inds = (self.dones_inds[1], self.dones_inds[1] + 1)

        # Run on server side
        print("PARAMS: ", config['db_host'], config['db_port'])
        self.r = redis.Redis(db=0, host=config['db_host'], port=config['db_port'])
        if config['pretrain'] is not None:
            print(f"Using existing replay buffer with {self.r.dbsize()} replays.")

    def add(self, obs_t, action, reward, obs_tp1, done, gamma):
        data = np.concatenate([obs_t, action, [reward], obs_tp1, [done], [gamma]], axis=0)
        self._add_array(data)

    def sample(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        gammas = []

        # Warning: may be too slow
        keys = [self.r.randomkey() for _ in range(int(batch_size))]
        replays = self.r.mget(keys)

        for replay in replays:
            replay = self._convert_string(replay)

            states.append(replay[self.state_inds[0]:self.state_inds[1]])
            actions.append(replay[self.action_inds[0]:self.action_inds[1]])
            rewards.append(replay[self.reward_inds[0]:self.reward_inds[1]])
            next_states.append(replay[self.next_state_inds[0]:self.next_state_inds[1]])
            dones.append(replay[self.dones_inds[0]:self.dones_inds[1]])
            gammas.append(replay[self.gammas_inds[0]:self.gammas_inds[1]])

        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards).reshape([batch_size, 1])
        next_states = np.asarray(next_states)
        dones = np.asarray(dones).reshape([batch_size, 1])
        gammas = np.asarray(gammas).reshape([batch_size, 1])

        return states, actions, rewards, next_states, dones, gammas

    def _add_array(self, d):
        d = d.astype('float16').tostring()
        key = str(self.r.dbsize())
        self.r.set(key, d)

    def _convert_string(self, A1):
        return np.fromstring(A1, dtype='float16')

    def __len__(self):
        return self.r.dbsize()


def create_replay_buffer(config):
    #size = config['replay_mem_size']
    #if config['replay_memory_prioritized']:
    #    alpha = config['priority_alpha']
    #    return PrioritizedReplayBuffer(size=size, alpha=alpha)
    #return ReplayBuffer(size)
    return RedisReplayBuffer(config)