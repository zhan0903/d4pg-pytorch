import abc
import copy
from collections import OrderedDict
from itertools import cycle
import numpy as np


class FeaturesBase(metaclass=abc.ABCMeta):
    def __init__(self):
        self.left_indices = None
        self.right_indices = None
        self.names = None

    def get_left_indices(self):
        """Return indices of left parts. """
        self.get_names()
        if self.left_indices is None:
            self._init_complement_indices()
        return self.left_indices

    def get_right_indices(self):
        """Return indices of right parts. """
        self.get_names()
        if self.left_indices is None:
            self._init_complement_indices()
        return self.right_indices

    def _init_complement_indices(self):
        """Fill self.left_indices and self.right_indices from self.names"""
        self.left_indices = []
        self.right_indices = []

        for index, name in enumerate(self.names):
            if name[-1] == 'l':
                try:
                    right_index = self.names.index(name[:-1] + 'r')
                    self.left_indices.append(index)
                    self.right_indices.append(right_index)
                except ValueError:
                    continue
        self.left_indices = np.array(self.left_indices)
        self.right_indices = np.array(self.right_indices)

    def get_names(self):
        """Return names of body parts and create them if necessary. """
        if not self.names:
            self._create_names()
        return self.names

    @abc.abstractmethod
    def _create_names(self):
        pass

    @abc.abstractmethod
    def to_numpy(self):
        pass


class Plainer(FeaturesBase):
    """Construct numpy array from given raw dict. """

    def __init__(self, raw_data, prefix=''):
        super(Plainer, self).__init__()

        self.raw_data = copy.deepcopy(raw_data)
        self.prefix = prefix
        self._create_names()

        for raw_part in self.raw_data:
            self.raw_data[raw_part] = np.array(self.raw_data[raw_part])

    def to_numpy(self):
        """Return numpy array from self.raw_data. """
        arrays = [self.raw_data[raw_part] for raw_part in sorted(self.raw_data.keys())]
        return np.concatenate(arrays)

    def _create_names(self):
        """Create self.names from raw data dict. """
        self.names = []
        for raw_part in sorted(self.raw_data.keys()):
            for _, coor_name in zip(range(len(self.raw_data[raw_part])), cycle(('x', 'y', 'z'))):
                self.names.append('{}_{}_{}'.format(self.prefix, coor_name, raw_part))


class Forces(Plainer):
    """Construct and process features of forces. """

    def __init__(self, raw_data, prefix=''):
        super(Plainer, self).__init__()

        self.raw_data = copy.deepcopy(raw_data)
        self.prefix = prefix
        self._create_names()

        for raw_part in self.raw_data:
            self.raw_data[raw_part] = np.array(self.raw_data[raw_part])

    def _create_names(self):
        self.names = []
        for raw_part in sorted(self.raw_data.keys()):
            for i in range(len(self.raw_data[raw_part])):
                self.names.append('{}_{}'.format(i, raw_part))


class RelativePlainer(FeaturesBase):
    """Construct relative features from absolute ones. """

    def __init__(self, raw_body_pos, basis):
        super(RelativePlainer, self).__init__()

        self.body_pos = copy.deepcopy(raw_body_pos)

        self.basis = basis
        self.body_parts = sorted(self.body_pos.keys())

        basis_pos = np.array(self.body_pos[basis])

        for body_part in self.body_parts:
            self.body_pos[body_part] = np.array(self.body_pos[body_part])
            self.body_pos[body_part] -= basis_pos

    def _create_names(self):
        self.names = []
        for body_part in self.body_parts:
            for _, coor_name in zip(range(len(self.body_pos[body_part])), cycle(('x', 'y', 'z'))):
                self.names.append('{}_rel_to_{}_{}'.format(coor_name, self.basis, body_part))

    def to_numpy(self):
        arrays = [self.body_pos[body_part] for body_part in self.body_parts]
        return np.concatenate(arrays)


class Activations(FeaturesBase):
    """Construct muscles activation features. """

    def __init__(self, observation):
        super(Activations, self).__init__()
        self.raw_muscles = copy.deepcopy(observation['muscles'])

        muscle_names = sorted(self.raw_muscles.keys())
        self.muscles = []
        self._create_names()

        for muscle_name in muscle_names:
            muscle_features = self.raw_muscles[muscle_name]
            # feature_names = sorted(muscle_features.keys())
            self.muscles.append(muscle_features['activation'])
            self.names.append('activation_{}'.format(muscle_name))

    def to_numpy(self):
        return np.array(self.muscles)

    def _create_names(self):
        self.names = []


class Muscles(FeaturesBase):
    def __init__(self, raw_muscles):
        super(Muscles, self).__init__()
        self.raw_muscles = copy.deepcopy(raw_muscles)

        muscle_names = sorted(raw_muscles.keys())
        self.muscles = []
        self._create_names()

        for muscle_name in muscle_names:
            muscle_features = raw_muscles[muscle_name]
            # feature_names = sorted(muscle_features.keys())
            feature_names = ('activation', 'fiber_length', 'fiber_force')
            for feature_name in feature_names:
                if feature_name == 'fiber_force':
                    self.muscles.append(muscle_features[feature_name])
                else:
                    self.muscles.append(muscle_features[feature_name])
                self.names.append('{}_{}'.format(feature_name, muscle_name))

    def to_numpy(self):
        return np.array(self.muscles)

    def _create_names(self):
        self.names = []


class Heights(FeaturesBase):
    def __init__(self, observation):
        super(Heights, self).__init__()
        self.heights = []
        self._create_names()

        feature_names = ('body_pos', 'body_vel', 'body_acc')

        body_parts = sorted(observation[feature_names[0]].keys())

        for feature_name in feature_names:
            feature = observation[feature_name]
            for body_part in body_parts:
                self.heights.append(feature[body_part][1])
                self.names.append('height_{}_{}'.format(feature_name, body_part))

    def _create_names(self):
        self.names = []

    def to_numpy(self):
        return np.array(self.heights)


class MassCenter(FeaturesBase):
    def __init__(self, relative_against):
        super(MassCenter, self).__init__()

        self.relative_against = relative_against
        self.relatives = {
            "vel": OrderedDict(),
            "pos": OrderedDict(),
            "acc": OrderedDict()
        }

        self._create_names()

    def update(self, observation):
        for feature in ("vel", "pos", "acc"):
            for body_part in self.relative_against:
                limb = np.array(observation['body_{}'.format(feature)][body_part])
                mass_center = np.array(observation['misc']['mass_center_{}'.format(feature)])
                self.relatives[feature][body_part] = limb - mass_center

    def _create_names(self):
        self.names = []

        coors = ('x', 'y', 'z')
        features = ('pos', 'vel', 'acc')

        for body_part in self.relative_against:
            for feature in features:
                for coor in coors:
                    self.names.append('mass_center_{}_{}_{}'.format(body_part, feature, coor))

    def to_numpy(self):
        if len(self.relative_against) == 0:
            return np.empty((0,))

        features = []

        for body_part in self.relative_against:
            for feature in ("pos", "vel", "acc"):
                features.append(self.relatives[feature][body_part])

        return np.concatenate(features)


class Relatives(FeaturesBase):
    def __init__(self, relatives_type, relative_against, features=('pos', 'vel', 'acc')):
        """
        Construct relative features from absolute ones.

        Args:
            relatives_type (str): name of the functional (body|body_rot|...) feature-set, used in names
            relative_against (list): names of features relative against
            features (tuple): types of features to process ('pos', 'vel', 'acc')
        """
        super(Relatives, self).__init__()
        self.relatives_type = relatives_type
        self.features = features
        self.relative_against = sorted(relative_against)
        self.per_type_names = []
        self.relatives = []

    def to_numpy(self):
        return np.concatenate(self.relatives)

    def update(self, observation):
        relatives = self._compute_relatives(observation)
        relatives = self._flatten_features(relatives)
        self.relatives = relatives

    def _create_names(self):
        self.names = []
        for relative_type in self.features:
            relative_type_feature_names = ['{}_{}_{}'.format(self.relatives_type, relative_type, name)
                                           for name in self.per_type_names]
            self.names += relative_type_feature_names

    def _compute_relatives(self, observation):
        result = {
            'pos': OrderedDict(),
            'vel': OrderedDict(),
            'acc': OrderedDict()
        }

        for feature in ('pos', 'vel', 'acc'):
            for body_part in self.relative_against:
                result[feature][body_part] = RelativePlainer(observation[self.relatives_type.format(feature)],
                                                             basis=body_part).to_numpy()
        result = [result[relative_type] for relative_type in self.features]

        return result

    @staticmethod
    def _flatten_features(relatives):

        def plain_feature_dict(feature_dict):
            features = [feature_dict[feature] for feature in feature_dict]
            return features

        processed_relatives = [plain_feature_dict(relative) for relative in relatives]
        flatten_relatives = []

        for processed_relative in processed_relatives:
            flatten_relatives += processed_relative

        return flatten_relatives
