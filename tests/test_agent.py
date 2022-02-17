import numpy as np

import tensorflow as tf
from core.agent import RLAgent, _encode_action, _calculate_log_proba_of_action
from core.environment import SpinEnvironment
from core.lattice import KagomeLattice


def test_rl_agent_has_correct_action_space():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation = environment.reset()

    agent = RLAgent(lattice)

    agent_action_index = agent.act(observation)
    assert all(np.unique(agent_action_index) == [0, 1])


def test_rl_agent_policy_is_on_gpu():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation = environment.reset()

    agent = RLAgent(lattice)
    # Weights are initialized after call
    agent_action_index = agent.act(observation)

    assert all(
        ["GPU" in weight.device for weight in agent.policy_network.weights]
    )


def test_rl_agent_policy_weights_have_correct_dtype():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation = environment.reset()

    agent = RLAgent(lattice)
    # Weights are initialized after call
    agent_action_index = agent.act(observation)

    assert all(
        [weight.dtype == tf.float32 for weight in agent.policy_network.weights]
    )


def test_rl_agent_has_same_lattice_adjacency_with_environment():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)

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


def test_agent_actions_are_properly_encoded():
    agent_action_index = tf.constant(
        [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1]],
        dtype=tf.int64,
    )
    encoded_action = _encode_action(agent_action_index)
    expected = tf.constant(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=tf.float32,
    )

    tf.debugging.assert_equal(encoded_action, expected)


def test_calculate_correct_log_proba_for_agent_action():
    agent_action_index = tf.constant(
        [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1]],
        dtype=tf.int64,
    )
    agent_logits = tf.constant(
        [
            [-1.1, 2.3],
            [0.3, 0.0],
            [3.5, 6.3],
            [-0.4, -8.1],
            [0.0, 0.0],
            [0.0, -6.5],
            [4.2, -5.8],
            [-4.7, 3.9],
            [4.1, 4.1],
            [-5.8, -5.8],
            [-1.5, 1.5],
            [-8.5, 7.1],
        ],
        dtype=tf.float32,
    )

    action_log_proba = _calculate_log_proba_of_action(
        agent_logits, agent_action_index
    )

    probas = tf.nn.softmax(agent_logits)
    encoded_action = _encode_action(agent_action_index)
    expected = tf.reduce_sum(
        tf.math.multiply(tf.math.log(probas), encoded_action)
    )

    tf.debugging.assert_near(action_log_proba, expected)



def test_same_agent_action_maps_state_back():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation_0 = environment.reset()

    agent = RLAgent(lattice)
    agent_action_index = agent.act(observation_0)
    observation_1, _, _ = environment.step(agent_action_index)
    assert any(tf.math.not_equal(environment.spin_state, observation_0))

    # Same action maps observation_1 back to observation_0
    observation_2, _, _ = environment.step(agent_action_index)

    tf.debugging.assert_equal(observation_2, observation_0)

