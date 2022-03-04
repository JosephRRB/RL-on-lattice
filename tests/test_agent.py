import dgl
import numpy as np

import tensorflow as tf
from core.agent import (
    RLAgent,
)
from core.environment import SpinEnvironment
from core.lattice import KagomeLattice
from core.policy_network import _create_batched_graphs, GraphPolicyNetwork


def test_policy_network_has_correct_output_shape():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    n_nodes = lattice.num_nodes()
    spin_state = tf.constant(
        [[1], [1], [1], [-1], [1], [-1], [-1], [-1], [-1], [-1], [1], [-1]],
        dtype=tf.float32,
    )
    batched_graphs = _create_batched_graphs(lattice, n_batch=2)
    batched_obs = tf.concat([spin_state, spin_state], axis=0)

    policy_network = GraphPolicyNetwork(n_hidden=10, n_nodes=n_nodes)
    logits1, logits2 = policy_network(batched_graphs, batched_obs)
    assert logits1.shape == (2, n_nodes)
    assert logits2.shape == (2, n_nodes)


def test_policy_network_gives_same_logits_for_same_state_inputs():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    n_nodes = lattice.num_nodes()
    spin_state = tf.constant(
        [[1], [1], [1], [-1], [1], [-1], [-1], [-1], [-1], [-1], [1], [-1]],
        dtype=tf.float32,
    )
    batched_graphs = _create_batched_graphs(lattice, n_batch=2)
    batched_obs = tf.concat([spin_state, spin_state], axis=0)

    policy_network = GraphPolicyNetwork(n_hidden=10, n_nodes=n_nodes)
    logits1, logits2 = policy_network(batched_graphs, batched_obs)

    tf.debugging.assert_equal(logits1[0, :], logits1[1, :])
    tf.debugging.assert_equal(logits2[0, :], logits2[1, :])


def test_rl_agent_selects_no_more_than_the_number_of_lattice_nodes():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation = environment.reset()

    agent = RLAgent(lattice)
    selected_nodes = agent.act(observation)
    assert selected_nodes.shape[1] <= lattice.num_nodes()


def test_rl_agent_does_not_select_nodes_more_than_once():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation = environment.reset()

    agent = RLAgent(lattice)
    selected_nodes = agent.act(observation)

    _, _, counts = tf.unique_with_counts(
        tf.reshape(selected_nodes, shape=(-1,))
    )
    tf.debugging.assert_equal(counts, 1)


def test_rl_agent_acts_by_selecting_node_indices():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation = environment.reset()

    agent = RLAgent(lattice)
    selected_nodes = agent.act(observation)

    assert selected_nodes.dtype == tf.int32
    tf.debugging.assert_greater_equal(selected_nodes, 0)
    tf.debugging.assert_less(selected_nodes, lattice.num_nodes())


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
    agent = RLAgent(lattice)

    assert all(
        [weight.dtype == tf.float32 for weight in agent.policy_network.weights]
    )


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


def test_batched_graphs_has_same_adjacency_as_original():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    batch_graphs = _create_batched_graphs(lattice, n_batch=2)
    g1, g2 = dgl.unbatch(batch_graphs)

    adjacency_for_first = g1.adj(scipy_fmt="coo").todense()
    adjacency_for_second = g2.adj(scipy_fmt="coo").todense()
    adjacency_from_environment = lattice.adj(scipy_fmt="coo").todense()

    np.testing.assert_array_equal(
        adjacency_for_first, adjacency_from_environment
    )
    np.testing.assert_array_equal(
        adjacency_for_second, adjacency_from_environment
    )


#
# def test_agent_actions_are_properly_encoded():
#     agent_action_index = tf.constant(
#         [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1]],
#         dtype=tf.int64,
#     )
#     encoded_action = _encode_action(agent_action_index)
#     expected = tf.constant(
#         [
#             [1, 0],
#             [0, 1],
#             [1, 0],
#             [0, 1],
#             [1, 0],
#             [0, 1],
#             [1, 0],
#             [0, 1],
#             [1, 0],
#             [0, 1],
#             [1, 0],
#             [0, 1],
#         ],
#         dtype=tf.float32,
#     )
#
#     tf.debugging.assert_equal(encoded_action, expected)


def test_calculate_correct_log_probas_for_agent_selecting_only_one_node():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    agent = RLAgent(lattice)

    agent_selected_nodes = tf.ragged.constant(
        [[0], [11]],
        dtype=tf.int32,
    )
    # Selecting nodes #0 for the first batch and #11 for the second batch is
    # most likely
    node_logits = tf.constant(
        [[-0.1] + [-100.0] * 11, [-100.0] * 11 + [-0.1]],
        dtype=tf.float32,
    )
    # Selecting more than 1 node is very unlikely
    n_nodes_logits = tf.constant(
        [[-0.1] + [-100.0] * 11, [-0.1] + [-100.0] * 11],
        dtype=tf.float32,
    )
    log_probas_of_actions = agent._calculate_action_log_probas_from_logits(
        node_logits, n_nodes_logits, agent_selected_nodes
    )

    expected_log_probas = tf.constant([[0.0], [0.0]], dtype=tf.float32)
    tf.debugging.assert_near(log_probas_of_actions, expected_log_probas)


def test_calculate_correct_log_probas_for_agent_actions():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    agent = RLAgent(lattice)

    agent_selected_nodes = tf.ragged.constant(
        [[4, 7], [6]],
        dtype=tf.int32,
    )
    node_logits = tf.constant(
        [
            [
                0.2,
                0.1,
                0.0,
                -0.1,
                -0.2,
                -0.3,
                -0.4,
                -0.5,
                -0.6,
                -0.7,
                -0.8,
                -0.9,
            ],
            [
                0.2,
                0.1,
                0.0,
                -0.1,
                -0.2,
                -0.3,
                -0.4,
                -0.5,
                -0.6,
                -0.7,
                -0.8,
                -0.9,
            ],
        ],
        dtype=tf.float32,
    )
    n_nodes_logits = tf.constant(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ],
        dtype=tf.float32,
    )

    log_probas_of_actions = agent._calculate_action_log_probas_from_logits(
        node_logits, n_nodes_logits, agent_selected_nodes
    ).numpy()

    # ------------------------------------------------------------------------------------------------------------------
    node_probas = tf.nn.softmax(node_logits).numpy()
    n_nodes_probas = tf.nn.softmax(n_nodes_logits).numpy()

    expected_action1_lp = np.log(
        node_probas[0, 4]
        * node_probas[0, 7]
        / (1 - node_probas[0, 4])
        * n_nodes_probas[0, 1]
    )
    expected_action2_lp = np.log(node_probas[1, 6] * n_nodes_probas[1, 0])

    tol = 1e-3
    assert np.abs(log_probas_of_actions[0, 0] - expected_action1_lp) < tol
    assert np.abs(log_probas_of_actions[1, 0] - expected_action2_lp) < tol


def test_agent_log_probas_of_actions_have_correct_shape():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    agent = RLAgent(lattice)

    spin_state = tf.constant(
        [[1], [1], [1], [-1], [1], [-1], [-1], [-1], [-1], [-1], [1], [-1]],
        dtype=tf.float32,
    )
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
        ],
        dtype=tf.int64,
    )
    batched_graphs = _create_batched_graphs(agent.graph, n_batch=2)
    batched_obs = tf.concat([spin_state, spin_state], axis=0)
    batched_action_indices = tf.concat(
        [agent_action_index, agent_action_index], axis=0
    )

    log_proba = agent.calculate_log_probas_of_agent_actions(
        agent.graph, spin_state, agent_action_index
    )
    batch_log_probas = agent.calculate_log_probas_of_agent_actions(
        batched_graphs, batched_obs, batched_action_indices
    )

    assert log_proba.shape == (1, 1)
    assert batch_log_probas.shape == (2, 1)


def test_same_agent_action_maps_state_back():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation_0 = environment.reset()

    agent = RLAgent(lattice)
    agent_action_index = agent.act(observation_0)
    observation_1, _ = environment.step(agent_action_index)
    assert any(tf.math.not_equal(environment.spin_state, observation_0))

    # Same action maps observation_1 back to observation_0
    observation_2, _ = environment.step(agent_action_index)

    tf.debugging.assert_equal(observation_2, observation_0)
