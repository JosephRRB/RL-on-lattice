import numpy as np

import tensorflow as tf
from core.agent import RLAgent
from core.environment import KagomeLatticeEnv


def test_rl_agent_has_correct_action_space():
    environment = KagomeLatticeEnv(n_sq_cells=2)
    lattice = environment.lattice
    observation = environment.reset()

    agent = RLAgent(lattice)

    agent_action_index = agent.act(observation)
    assert all(np.unique(agent_action_index) == [0, 1])


def test_rl_agent_policy_is_on_gpu():
    environment = KagomeLatticeEnv(n_sq_cells=2)
    lattice = environment.lattice
    observation = environment.reset()

    agent = RLAgent(lattice)
    # Weights are initialized after call
    agent_action_index = agent.act(observation)

    assert all(
        ["GPU" in weight.device for weight in agent.policy_network.weights]
    )


def test_rl_agent_policy_weights_have_correct_dtype():
    environment = KagomeLatticeEnv(n_sq_cells=2)
    lattice = environment.lattice
    observation = environment.reset()

    agent = RLAgent(lattice)
    # Weights are initialized after call
    agent_action_index = agent.act(observation)

    assert all(
        [weight.dtype == tf.float32 for weight in agent.policy_network.weights]
    )


def test_rl_agent_has_correct_lattice_adjacency():
    environment = KagomeLatticeEnv(n_sq_cells=2)
    lattice = environment.lattice

    agent = RLAgent(lattice)

    adjacency_from_agent = agent.policy_network.graph.adj(
        scipy_fmt="coo"
    ).todense()
    adjacency_from_environment = environment.lattice.adj(
        scipy_fmt="coo"
    ).todense()

    np.testing.assert_array_equal(
        adjacency_from_agent, adjacency_from_environment
    )


# test trainable
