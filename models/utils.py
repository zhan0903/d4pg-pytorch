import random
import gym
import numpy as np
import matplotlib.pyplot as plt

from .d3pg import PolicyNetwork as PolicyNetworkDDPG
from .d4pg import PolicyNetwork as PolicyNetworkD4PG

from .d3pg import LearnerD3PG
from .d4pg import LearnerD4PG

def create_actor(model_name, num_actions, num_inputs, hidden_size):
    model_name = model_name.lower()
    if model_name == "d3pg":
        return PolicyNetworkDDPG(num_inputs=num_inputs, num_actions=num_actions, hidden_size=hidden_size)
    elif model_name == "d4pg":
        return PolicyNetworkD4PG(num_inputs=num_inputs, num_actions=num_actions, hidden_size=hidden_size)
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def create_learner(config, batch_queue):
    model_name = config["model"].lower()
    if model_name == "d3pg":
        return LearnerD3PG(config, batch_queue)
    elif model_name == "d4pg":
        return LearnerD4PG(config, batch_queue)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
