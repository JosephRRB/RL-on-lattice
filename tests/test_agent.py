import numpy as np

import dgl
import tensorflow as tf
from core.agent import (
    RLAgent,
    _encode_action,
    _batch_calculate_log_proba_of_actions,
)
from core.environment import SpinEnvironment
from core.lattice import KagomeLattice


def test_rl_agent_has_correct_action_space():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation = environment.reset()

    agent = RLAgent(lattice)

    agent_action_index = agent.act(observation)
    assert all(np.unique(agent_action_index) == [0, 1])


# def test_rl_agent_policy_is_on_gpu():
#     lattice = KagomeLattice(n_sq_cells=2).lattice
#     environment = SpinEnvironment(lattice)
#     observation = environment.reset()
#
#     agent = RLAgent(lattice)
#     # Weights are initialized after call
#     _ = agent.act(observation)
#
#     assert all(
#         ["GPU" in weight.device for weight in agent.policy_network.weights]
#     )


def test_rl_agent_policy_weights_have_correct_dtype():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation = environment.reset()

    agent = RLAgent(lattice)
    # Weights are initialized after call
    _ = agent.act(observation)

    assert all(
        [weight.dtype == tf.float32 for weight in agent.policy_network.weights]
    )


def test_rl_agent_action_has_correct_shape():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation = environment.reset()

    agent = RLAgent(lattice)
    agent_action_index = agent.act(observation)

    assert agent_action_index.shape == observation.shape


def test_rl_agent_has_same_lattice_adjacency_with_environment():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)

    agent = RLAgent(lattice)

    adjacency_from_agent = agent.graph.adj(scipy_fmt="coo").todense()
    adjacency_from_environment = environment.lattice.adj(
        scipy_fmt="coo"
    ).todense()

    np.testing.assert_array_equal(
        adjacency_from_agent, adjacency_from_environment
    )


def test_rl_agent_batched_graphs_has_same_adjacency_as_original():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)

    agent = RLAgent(lattice, n_batch=2)

    g1, g2 = dgl.unbatch(agent.batch_graphs)

    adjacency_for_first = g1.adj(scipy_fmt="coo").todense()
    adjacency_for_second = g2.adj(scipy_fmt="coo").todense()
    adjacency_from_environment = environment.lattice.adj(
        scipy_fmt="coo"
    ).todense()

    np.testing.assert_array_equal(
        adjacency_for_first, adjacency_from_environment
    )
    np.testing.assert_array_equal(
        adjacency_for_second, adjacency_from_environment
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
        [
            [0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [1],
            [0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [0],
        ],
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

    log_proba_of_actions = _batch_calculate_log_proba_of_actions(
        agent_logits, agent_action_index, n_batch=2
    )

    probas1 = tf.nn.softmax(agent_logits[:12, :])
    encoded_action1 = _encode_action(agent_action_index[:12, :])
    expected1 = tf.reduce_sum(
        tf.math.multiply(tf.math.log(probas1), encoded_action1)
    )

    tf.debugging.assert_near(log_proba_of_actions[0, 0], expected1)

    probas2 = tf.nn.softmax(agent_logits[12:, :])
    encoded_action2 = _encode_action(agent_action_index[12:, :])
    expected2 = tf.reduce_sum(
        tf.math.multiply(tf.math.log(probas2), encoded_action2)
    )

    tf.debugging.assert_near(log_proba_of_actions[1, 0], expected2)


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
