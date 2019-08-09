import gym
#from utils.utils import NormalizedActions

from .pendulum import PendulumWrapper
from .bipedal import BipedalWalker
from .lunar_lander_continous import LunarLanderContinous
from .learn_to_move import LearnToMove


def create_env_wrapper(config):
    env = config['env'].lower()
    if env == "pendulum-v0":
        return PendulumWrapper(config)
    elif env == "bipedalwalker-v2":
        return BipedalWalker(config)
    elif env == "lunarlandercontinuous-v2":
        return LunarLanderContinous(config)
    elif env == "learntomove":
        return LearnToMove(config)
    else:
        raise ValueError("Unknown environment.")