import dgl
import numpy as np

import tensorflow as tf
from core.agent import (
    RLAgent,
    _calculate_action_log_probas_from_logits,
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


def test_reversed_agent_action_maps_state_back():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation_0 = environment.reset()

    agent = RLAgent(lattice)
    selected_nodes = agent.act(observation_0)
    observation_1, _ = environment.step(selected_nodes)
    assert any(tf.math.not_equal(environment.spin_state, observation_0))

    # Reversed selected nodes maps observation_1 back to observation_0
    reversed_selected_nodes = selected_nodes[:, ::-1]
    observation_2, _ = environment.step(reversed_selected_nodes)

    tf.debugging.assert_equal(observation_2, observation_0)

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


def test_calculate_correct_log_probas_for_agent_actions():
    agent_selected_nodes = tf.ragged.constant(
        [[4, 1, 9], [6]],
        dtype=tf.int32,
    )
    node_logits = tf.random.uniform(shape=(2, 12))
    n_nodes_logits = tf.random.uniform(shape=(2, 12))

    log_probas_of_actions = _calculate_action_log_probas_from_logits(
        node_logits, n_nodes_logits, agent_selected_nodes
    ).numpy()

    # ------------------------------------------------------------------------------------------------------------------
    node_probas = tf.nn.softmax(node_logits).numpy()
    n_nodes_probas = tf.nn.softmax(n_nodes_logits).numpy()

    expected_action1_lp = np.log(
        node_probas[0, 4]
        * node_probas[0, 1]
        / (1 - node_probas[0, 4])
        * node_probas[0, 9]
        / (1 - node_probas[0, 4] - node_probas[0, 1])
        * n_nodes_probas[0, 2]
    )
    expected_action2_lp = np.log(node_probas[1, 6] * n_nodes_probas[1, 0])

    tol = 1e-3
    assert np.abs(log_probas_of_actions[0, 0] - expected_action1_lp) < tol
    assert np.abs(log_probas_of_actions[1, 0] - expected_action2_lp) < tol


def test_calculated_log_proba_remains_finite():
    agent_selected_nodes = tf.ragged.constant(
        [[0, 9, 7]],
        dtype=tf.int32,
    )
    # First chosen node is highly likely, while others are equally very unlikely
    # eps = 1.7e-7
    # eps = 1.64e-7
    eps = 1.63e-7
    # eps = 1.6e-7
    node_probs = tf.constant([[1.0 - eps] + [eps / 11] * 11], dtype=tf.float32)
    node_logits = tf.math.log(node_probs)
    # # 3 nodes are very likely to be chosen
    # n_node_probs = tf.constant([
    #     [eps / 11, eps / 11, 1.0 - eps] + [eps / 11] * 9
    # ])
    # n_nodes_logits = tf.math.log(n_node_probs)
    n_nodes_logits = tf.random.uniform(shape=(1, 12))
    log_probas_of_actions = _calculate_action_log_probas_from_logits(
        node_logits, n_nodes_logits, agent_selected_nodes
    )
    assert tf.math.is_finite(log_probas_of_actions)

    # expected_p = tf.constant([[
    #     (1 - eps)*(1/11)*(1/10)
    # ]])
    # expected_lp = tf.math.log(expected_p)


def test_calculate_correct_log_probas_for_agent_selecting_only_one_node():
    agent_selected_nodes = tf.ragged.constant(
        [[0], [11], [6]],
        dtype=tf.int32,
    )
    node_logits = tf.random.uniform(shape=(3, 12))
    # Selecting only one node is most likely
    n_nodes_logits = tf.constant(
        [
            [10.0] + [-10.0] * 11,
            [10.0] + [-10.0] * 11,
            [10.0] + [-10.0] * 11,
        ]
    )
    log_probas_of_actions = _calculate_action_log_probas_from_logits(
        node_logits, n_nodes_logits, agent_selected_nodes
    )

    normed_logits = tf.nn.log_softmax(node_logits)
    expected_lp = tf.gather(
        normed_logits, agent_selected_nodes, axis=1, batch_dims=1
    ).to_tensor()

    tf.debugging.assert_equal(log_probas_of_actions, expected_lp)


def test_selecting_last_node_does_not_change_probability():
    agent_selected_nodes = tf.ragged.constant(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ],
        dtype=tf.int32,
    )
    node_logits = tf.random.uniform(shape=(1, 12))
    node_logits = tf.concat([node_logits, node_logits], axis=0)

    # Selecting N number of nodes are equally likely
    n_nodes_logits = tf.constant([[0.0] * 12, [0.0] * 12])

    log_probas_of_actions = _calculate_action_log_probas_from_logits(
        node_logits, n_nodes_logits, agent_selected_nodes
    )

    tf.debugging.assert_equal(
        log_probas_of_actions[0, :], log_probas_of_actions[1, :]
    )


def test_agent_log_probas_of_actions_have_correct_shape():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    agent = RLAgent(lattice)

    spin_state = tf.constant(
        [[1], [1], [1], [-1], [1], [-1], [-1], [-1], [-1], [-1], [1], [-1]],
        dtype=tf.float32,
    )
    batched_graphs = _create_batched_graphs(agent.graph, n_batch=2)
    batched_obs = tf.concat([spin_state, spin_state], axis=0)

    agent_selected_nodes_1 = tf.ragged.constant(
        [[4, 1, 9, 7, 3]],
        dtype=tf.int32,
    )
    agent_selected_nodes_2 = tf.ragged.constant(
        [[5, 4, 8, 1, 2, 9], [6]],
        dtype=tf.int32,
    )

    log_proba = agent.calculate_log_probas_of_agent_actions(
        agent.graph, spin_state, agent_selected_nodes_1
    )
    batch_log_probas = agent.calculate_log_probas_of_agent_actions(
        batched_graphs, batched_obs, agent_selected_nodes_2
    )

    assert log_proba.shape == (1, 1)
    assert batch_log_probas.shape == (2, 1)

