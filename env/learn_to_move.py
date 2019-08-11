import numpy as np
from osim.env import L2M2019Env

from .learn_to_move_features import Relatives, Muscles, Activations, MassCenter, Plainer, Forces, Heights


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


class ExtendedObservationTransformer:
    """Transforms observation signal. """

    def __init__(self):
        self.features = {
            "body_pos_relative": ["pelvis", "torso", "head"],
            "body_rot_relative": ["pelvis"],
            "mass_center_relative": ["pelvis"],
            "raw_features": ["forces"],
            "raw_height": ["pelvis", "torso", "head"],
            "raw_vel": ["pelvis", "torso"],
            "raw_rot": ["pelvis"]
        }

        self.pos_relative_features = Relatives(relatives_type='body_{}',
                                               relative_against=self.features.get("body_pos_relative", []))

        self.rot_relative_features = Relatives(relatives_type='body_{}_rot',
                                               relative_against=self.features.get("body_rot_relative", []))

        self.mass_center_relative = MassCenter(self.features.get("mass_center_relative", []))

        self.left_indices = None
        self.right_indices = None

    def transform(self, observation):
        self.pos_relative_features.update(observation)
        self.rot_relative_features.update(observation)
        self.mass_center_relative.update(observation)

        pos_relatives = self.pos_relative_features.to_numpy()
        rot_relatives = self.rot_relative_features.to_numpy()
        mass_center_relatives = self.mass_center_relative.to_numpy()

        raw_features = []
        for raw_feature in sorted(self.features.get('raw_features', [])):
            if raw_feature == 'forces':
                forces = Forces(observation[raw_feature])
                raw_features.append(forces.to_numpy())
            elif raw_feature == 'muscles':
                muscles = Muscles(observation[raw_feature])
                raw_features.append(muscles.to_numpy())
            elif raw_feature == 'heights':
                heights = Heights(observation)
                raw_features.append(heights.to_numpy())
            elif raw_feature == 'activations':
                activations = Activations(observation)
                raw_features.append(activations.to_numpy())
            else:
                raw_feature = Plainer(observation[raw_feature])
                raw_features.append(raw_feature.to_numpy())

        # Add raw target velocity field
        target_v_field = observation['v_tgt_field'] # [2 x 11 x 11]
        raw_features.append(list(target_v_field.flatten()))

        for feature in self.features.get("raw_height"):
            raw_features.append(np.array([observation["body_pos"][feature][1]]))

        for feature in self.features.get("raw_vel"):
            raw_features.append(np.array(observation["body_vel"][feature]))
            raw_features.append(np.array(observation["body_acc"][feature]))

        for feature in self.features.get("raw_rot"):
            raw_features.append(np.array(observation["body_pos_rot"][feature]))
            raw_features.append(np.array(observation["body_vel_rot"][feature]))
            raw_features.append(np.array(observation["body_acc_rot"][feature]))

        cur_observation = np.concatenate([pos_relatives,
                                          rot_relatives,
                                          mass_center_relatives,
                                          *raw_features])

        return cur_observation.astype(np.float32)


class LearnToMove(object):
    def __init__(self, config):
        self.config = config
        self.env = L2M2019Env(visualize=bool(config['visualize']))
        self.project = True # False - dict of size 14, True - dict of size 4
        self.env.reset(project=self.project)

        self.observation_transformer = ObservationTransformer()
        #self.observation_transformer = ExtendedObservationTransformer()

    def step(self, action):
        obs, reward, done, _ = self.env.step(action.flatten(), project=self.project)
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
        return self.observation_transformer.transform(self.env.reset(project=self.project))

    def render(self):
        pass