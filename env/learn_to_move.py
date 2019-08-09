import numpy as np
from osim.env import L2M2019Env


class ObservationTransformer(object):
    """Transforms observation signal. """

    def __init__(self):
        pass

    def transform(self, observation):
        features = []

        features += [observation['pelvis']['height']]
        features += [observation['pelvis']['pitch']]
        features += [observation['pelvis']['roll']]
        features += observation['pelvis']['vel']

        for leg in ['l_leg', 'r_leg']:
            features += observation[leg]['ground_reaction_forces']
            features += [observation[leg]['joint']['hip_abd'],
                         observation[leg]['joint']['hip'],
                         observation[leg]['joint']['knee'],
                         observation[leg]['joint']['ankle']]
            features += [observation[leg]['d_joint']['hip_abd'],
                         observation[leg]['d_joint']['hip'],
                         observation[leg]['d_joint']['knee'],
                         observation[leg]['d_joint']['ankle']]
            features += [observation[leg]['HAB'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['HAD'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['HFL'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['GLU'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['HAM'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['RF'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['VAS'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['BFSH'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['GAS'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['SOL'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['TA'][k] for k in ['f', 'l', 'v']]

        target_v_field = observation['v_tgt_field'].flatten() # [2 x 11 x 11]
        features += target_v_field.tolist()

        return np.asarray(features)


class LearnToMove(object):
    def __init__(self, config):
        self.config = config
        self.env = L2M2019Env(visualize=bool(config['visualize']))
        self.env.reset()

        self.observation_transformer = ObservationTransformer()

    def step(self, action):
        obs, reward, done, _ = self.env.step(action.flatten())
        obs = self.observation_transformer.transform(obs)
        return obs, reward, done

    def get_action_space(self):
        class ActionSpace(object):
            def __init__(self):
                self.low = 0
                self.high = 1
                self.shape = (22,)

        return ActionSpace()

    def normalise_state(self, state):
        return state

    def normalise_reward(self, reward):
        return reward

    def reset(self):
        return self.observation_transformer.transform(self.env.reset())

    def render(self):
        pass