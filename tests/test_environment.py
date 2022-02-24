import numpy as np

import tensorflow as tf
from core.environment import (
    SpinEnvironment,
    _calculate_entropy,
    _calculate_reward,
)
from core.lattice import KagomeLattice


def test_environment_gives_correct_spins():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation = environment.reset()

    assert all(np.unique(observation) == [-1, 1])


# def test_environment_spin_state_is_on_gpu():
#     lattice = KagomeLattice(n_sq_cells=2).lattice
#     environment = SpinEnvironment(lattice)
#     observation = environment.reset()
#     assert "GPU" in observation.device


def test_environment_reset_gives_correct_spin_dtypes():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation = environment.reset()

    assert observation.dtype == tf.float32


def test_environment_resets():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation_1 = environment.reset()
    observation_2 = environment.reset()

    assert any(tf.math.not_equal(observation_1, observation_2))


def test_environment_correctly_tracks_lattice_spins_between_resets():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    observation_1 = environment.reset()
    tf.debugging.assert_equal(observation_1, environment.spin_state)

    observation_2 = environment.reset()
    tf.debugging.assert_equal(observation_2, environment.spin_state)


def test_environment_correctly_flips_spins_based_on_agent_action():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    old_observation = environment.reset()

    agent_action_index = tf.constant(
        [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1]],
        dtype=tf.int64,
    )

    new_observation, _ = environment.step(agent_action_index)

    tf.debugging.assert_equal(new_observation[::2, 0], old_observation[::2, 0])
    tf.debugging.assert_equal(
        new_observation[1::2, 0], -old_observation[1::2, 0]
    )


def test_environment_correctly_tracks_lattice_spins_after_steps():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    _ = environment.reset()

    agent_action_index = tf.constant(
        [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1]],
        dtype=tf.int64,
    )

    new_observation, _ = environment.step(agent_action_index)
    tf.debugging.assert_equal(environment.spin_state, new_observation)

    agent_action_index2 = tf.constant(
        [[0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1]],
        dtype=tf.int64,
    )

    new_observation2, _ = environment.step(agent_action_index2)
    assert any(tf.math.not_equal(new_observation, new_observation2))
    tf.debugging.assert_equal(environment.spin_state, new_observation2)


def test_environment_step_gives_correct_spin_dtypes():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    _ = environment.reset()

    agent_action_index = tf.constant(
        [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1]],
        dtype=tf.int64,
    )

    new_observation, _ = environment.step(agent_action_index)
    assert new_observation.dtype == tf.float32


def test_completely_disordered_spin_state_gives_max_entropy():
    spin_count = tf.constant([50, 50], dtype=tf.float32)
    total_counts = tf.constant(100, dtype=tf.float32)
    expected = tf.math.log(tf.constant(2, dtype=tf.float32))

    entropy = _calculate_entropy(spin_count, total_counts)

    tf.debugging.assert_equal(entropy, expected)


def test_completely_ordered_spin_state_gives_zero_entropy():
    spin_count = tf.constant([0, 100], dtype=tf.float32)
    total_counts = tf.constant(100, dtype=tf.float32)
    expected = tf.constant(0, dtype=tf.float32)

    entropy = _calculate_entropy(spin_count, total_counts)

    tf.debugging.assert_equal(entropy, expected)


def test_joint_entropy_of_stat_independent_spin_states_is_sum_of_entropies():
    spin_count = tf.constant([[10, 15], [30, 45]], dtype=tf.float32)
    total_counts = tf.constant(100, dtype=tf.float32)

    spin_count_A = tf.reduce_sum(spin_count, axis=0)
    spin_count_B = tf.reduce_sum(spin_count, axis=1)

    joint_entropy = _calculate_entropy(spin_count, total_counts)
    entropy_A = _calculate_entropy(spin_count_A, total_counts)
    entropy_B = _calculate_entropy(spin_count_B, total_counts)

    tf.debugging.assert_equal(joint_entropy, entropy_A + entropy_B)


def test_joint_entropy_of_stat_dependent_spin_states_is_correct():
    spin_count = tf.constant([[50, 0], [25, 25]], dtype=tf.float32)
    total_counts = tf.constant(100, dtype=tf.float32)
    expected = 3 * tf.math.log(tf.constant(2, dtype=tf.float32)) / 2

    joint_entropy = _calculate_entropy(spin_count, total_counts)

    tf.debugging.assert_equal(joint_entropy, expected)


def test_reward_for_unchanged_spin_state_is_zero():
    spin_state = tf.constant(
        [[-1], [1], [-1], [-1], [-1], [1], [1], [1], [-1], [1], [1], [1]],
        dtype=tf.float32,
    )
    expected = tf.constant(0, dtype=tf.float32)

    reward = _calculate_reward(spin_state, spin_state)

    tf.debugging.assert_equal(reward, expected)


def test_reward_for_changing_all_spins_is_zero():
    old_spin_state = tf.constant(
        [[-1], [1], [-1], [-1], [-1], [1], [1], [1], [-1], [1], [1], [1]],
        dtype=tf.float32,
    )
    new_spin_state = tf.constant(
        [[1], [-1], [1], [1], [1], [-1], [-1], [-1], [1], [-1], [-1], [-1]],
        dtype=tf.float32,
    )
    expected = tf.constant(0, dtype=tf.float32)

    reward = _calculate_reward(old_spin_state, new_spin_state)

    tf.debugging.assert_equal(reward, expected)


def test_reward_for_unchanged_completely_ordered_spin_state_is_zero():
    spin_state = tf.constant(
        [
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
        ],
        dtype=tf.float32,
    )
    expected = tf.constant(0, dtype=tf.float32)

    reward = _calculate_reward(spin_state, spin_state)

    tf.debugging.assert_equal(reward, expected)


def test_reward_for_changing_all_spins_for_completely_ordered_spin_state_is_zero():
    old_spin_state = tf.constant(
        [
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
        ],
        dtype=tf.float32,
    )
    new_spin_state = tf.constant(
        [
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
        ],
        dtype=tf.float32,
    )
    expected = tf.constant(0, dtype=tf.float32)

    reward = _calculate_reward(old_spin_state, new_spin_state)

    tf.debugging.assert_equal(reward, expected)


def test_reward_for_stat_independent_spin_states_is_one():
    old_spin_state = tf.constant(
        [[1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]],
        dtype=tf.float32,
    )
    new_spin_state = tf.constant(
        [[1], [1], [-1], [1], [1], [1], [1], [1], [1], [-1], [-1], [-1]],
        dtype=tf.float32,
    )
    expected = tf.constant(1, dtype=tf.float32)

    reward = _calculate_reward(old_spin_state, new_spin_state)

    tf.debugging.assert_near(reward, expected)


def test_environment_reward_is_clipped_close_to_but_not_zero():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    _ = environment.reset()

    agent_action_index = tf.constant(
        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
        dtype=tf.int64,
    )

    expected = tf.constant(0, dtype=tf.float32)
    _, reward = environment.step(agent_action_index)

    tf.debugging.assert_near(reward, expected)
    tf.debugging.assert_none_equal(reward, expected)


def test_environment_calculates_correct_batch_of_log_probas():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    env = SpinEnvironment(
        lattice,
        inverse_temp=0.5,
        spin_coupling=1.5,
        external_B=-0.25,
    )
    # # Put to GPU
    # spin_state = tf.cast(
    #     tf.constant(
    #         [[1], [1], [-1], [-1], [1], [1], [-1], [-1], [1], [1], [1], [-1]],
    #     ),
    #     dtype=tf.float32,
    # )
    spin_state1 = tf.constant(
        [[1], [1], [-1], [-1], [1], [1], [-1], [-1], [1], [1], [1], [-1]],
        dtype=tf.float32,
    )
    spin_state2 = tf.constant(
        [[-1], [-1], [1], [1], [-1], [-1], [1], [1], [-1], [-1], [-1], [1]],
        dtype=tf.float32,
    )
    spin_states = tf.concat([spin_state1, spin_state2], axis=0)

    log_probability = env.calculate_log_probas_of_spin_states(spin_states)

    n_edges_with_same_spin = 14
    n_edges_with_opposite_spin = 10
    total_spin_interaction = (
        n_edges_with_same_spin - n_edges_with_opposite_spin
    )
    total_spin = 7 - 5
    expected_log_probability1 = env.inverse_temp * (
        env.spin_coupling * total_spin_interaction
        + env.external_B * total_spin
    )
    expected_log_probability2 = env.inverse_temp * (
            env.spin_coupling * total_spin_interaction
            + env.external_B * (-total_spin)
    )
    expected_log_probability = tf.constant([[expected_log_probability1], [expected_log_probability2]], dtype=tf.float32)
    tf.debugging.assert_equal(log_probability, expected_log_probability)


def test_environment_lattice_graph_does_not_have_features_after_log_proba():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    env = SpinEnvironment(
        lattice,
        inverse_temp=0.5,
        spin_coupling=1.5,
        external_B=-0.25,
    )
    spin_state = env.reset()
    _ = env.calculate_log_probas_of_spin_states(spin_state)

    assert env.lattice.ndata == dict()


def test_aligning_spin_state_is_more_likely_if_ferromagnetic():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    # set external_B to 0
    env = SpinEnvironment(
        lattice,
        inverse_temp=1,
        spin_coupling=1,
        external_B=0,
    )
    # Put to GPU
    # Node 8 is misaligned with respect to its nearest neighbors
    # old_spin_state = tf.cast(
    #     tf.constant(
    #         [[1], [1], [1], [-1], [1], [-1], [-1], [-1], [1], [-1], [1], [-1]],
    #     ),
    #     dtype=tf.float32,
    # )
    old_spin_state = tf.constant(
        [[1], [1], [1], [-1], [1], [-1], [-1], [-1], [1], [-1], [1], [-1]],
        dtype=tf.float32,
    )
    # Flip node 8
    new_spin_state = tf.constant(
        [[1], [1], [1], [-1], [1], [-1], [-1], [-1], [-1], [-1], [1], [-1]],
        dtype=tf.float32,
    )

    old_log_proba = env.calculate_log_probas_of_spin_states(old_spin_state)
    new_log_proba = env.calculate_log_probas_of_spin_states(new_spin_state)
    delta_log_proba = new_log_proba - old_log_proba

    assert env.spin_coupling > 0
    assert delta_log_proba > 0


def test_misaligning_spin_state_is_less_likely_if_ferromagnetic():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    # set external_B to 0
    env = SpinEnvironment(
        lattice,
        inverse_temp=1,
        spin_coupling=1,
        external_B=0,
    )
    # Put to GPU
    # Node 8 is aligned with respect to its nearest neighbors
    # old_spin_state = tf.cast(
    #     tf.constant(
    #         [
    #             [1],
    #             [1],
    #             [1],
    #             [-1],
    #             [1],
    #             [-1],
    #             [-1],
    #             [-1],
    #             [-1],
    #             [-1],
    #             [1],
    #             [-1],
    #         ],
    #     ),
    #     dtype=tf.float32,
    # )
    old_spin_state = tf.constant(
            [
                [1],
                [1],
                [1],
                [-1],
                [1],
                [-1],
                [-1],
                [-1],
                [-1],
                [-1],
                [1],
                [-1],
            ],
        dtype=tf.float32,
        )

    # Flip node 8
    new_spin_state = tf.constant(
        [
            [1],
            [1],
            [1],
            [-1],
            [1],
            [-1],
            [-1],
            [-1],
            [1],
            [-1],
            [1],
            [-1],
        ],
        dtype=tf.float32,
    )

    old_log_proba = env.calculate_log_probas_of_spin_states(old_spin_state)
    new_log_proba = env.calculate_log_probas_of_spin_states(new_spin_state)
    delta_log_proba = new_log_proba - old_log_proba

    assert env.spin_coupling > 0
    assert delta_log_proba < 0


def test_aligning_spin_state_is_less_likely_if_antiferromagnetic():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    # set external_B to 0
    env = SpinEnvironment(
        lattice,
        inverse_temp=1,
        spin_coupling=-1,
        external_B=0,
    )
    # Put to GPU
    # Node 8 is misaligned with respect to its nearest neighbors
    # old_spin_state = tf.cast(
    #     tf.constant(
    #         [[1], [1], [1], [-1], [1], [-1], [-1], [-1], [1], [-1], [1], [-1]],
    #     ),
    #     dtype=tf.float32,
    # )
    old_spin_state = tf.constant(
        [[1], [1], [1], [-1], [1], [-1], [-1], [-1], [1], [-1], [1], [-1]],
        dtype=tf.float32,
    )

    # Flip node 8
    new_spin_state = tf.constant(
        [[1], [1], [1], [-1], [1], [-1], [-1], [-1], [-1], [-1], [1], [-1]],
        dtype=tf.float32,
    )

    old_log_proba = env.calculate_log_probas_of_spin_states(old_spin_state)
    new_log_proba = env.calculate_log_probas_of_spin_states(new_spin_state)
    delta_log_proba = new_log_proba - old_log_proba

    assert env.spin_coupling < 0
    assert delta_log_proba < 0


def test_misaligning_spin_state_is_more_likely_if_antiferromagnetic():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    # set external_B to 0
    env = SpinEnvironment(
        lattice,
        inverse_temp=1,
        spin_coupling=-1,
        external_B=0,
    )
    # Put to GPU
    # Node 8 is aligned with respect to its nearest neighbors
    # old_spin_state = tf.cast(
    #     tf.constant(
    #         [
    #             [1],
    #             [1],
    #             [1],
    #             [-1],
    #             [1],
    #             [-1],
    #             [-1],
    #             [-1],
    #             [-1],
    #             [-1],
    #             [1],
    #             [-1],
    #         ],
    #     ),
    #     dtype=tf.float32,
    # )
    old_spin_state = tf.constant(
        [
            [1],
            [1],
            [1],
            [-1],
            [1],
            [-1],
            [-1],
            [-1],
            [-1],
            [-1],
            [1],
            [-1],
        ],
        dtype=tf.float32,
    )

    # Flip node 8
    new_spin_state = tf.constant(
        [
            [1],
            [1],
            [1],
            [-1],
            [1],
            [-1],
            [-1],
            [-1],
            [1],
            [-1],
            [1],
            [-1],
        ],
        dtype=tf.float32,
    )
    old_log_proba = env.calculate_log_probas_of_spin_states(old_spin_state)
    new_log_proba = env.calculate_log_probas_of_spin_states(new_spin_state)
    delta_log_proba = new_log_proba - old_log_proba

    assert env.spin_coupling < 0
    assert delta_log_proba > 0


def test_aligning_spin_to_external_B_is_more_likely():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    # set spin_coupling to 0
    env = SpinEnvironment(
        lattice,
        inverse_temp=1,
        spin_coupling=0,
        external_B=-0.5,
    )
    B_direction = int(np.sign(env.external_B))

    # Put to GPU
    # Node 8 is misaligned with respect to the external magnetic field
    # old_spin_state = tf.cast(
    #     tf.constant(
    #         [
    #             [1],
    #             [1],
    #             [1],
    #             [-1],
    #             [1],
    #             [-1],
    #             [-1],
    #             [-1],
    #             [-B_direction],
    #             [-1],
    #             [1],
    #             [-1],
    #         ],
    #     ),
    #     dtype=tf.float32,
    # )
    old_spin_state = tf.constant(
        [
            [1],
            [1],
            [1],
            [-1],
            [1],
            [-1],
            [-1],
            [-1],
            [-B_direction],
            [-1],
            [1],
            [-1],
        ],
        dtype=tf.float32,
    )

    # Flip node 8
    new_spin_state = tf.constant(
        [
            [1],
            [1],
            [1],
            [-1],
            [1],
            [-1],
            [-1],
            [-1],
            [B_direction],
            [-1],
            [1],
            [-1],
        ],
        dtype=tf.float32,
    )

    old_log_proba = env.calculate_log_probas_of_spin_states(old_spin_state)
    new_log_proba = env.calculate_log_probas_of_spin_states(new_spin_state)
    delta_log_proba = new_log_proba - old_log_proba

    assert delta_log_proba > 0


def test_misaligning_spin_to_external_B_is_more_likely():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    # set spin_coupling to 0
    env = SpinEnvironment(
        lattice,
        inverse_temp=1,
        spin_coupling=0,
        external_B=-0.5,
    )
    B_direction = int(np.sign(env.external_B))

    # Put to GPU
    # Node 8 is aligned with respect to the external magnetic field
    # old_spin_state = tf.cast(
    #     tf.constant(
    #         [
    #             [1],
    #             [1],
    #             [1],
    #             [-1],
    #             [1],
    #             [-1],
    #             [-1],
    #             [-1],
    #             [B_direction],
    #             [-1],
    #             [1],
    #             [-1],
    #         ],
    #     ),
    #     dtype=tf.float32,
    # )
    old_spin_state = tf.constant(
        [
            [1],
            [1],
            [1],
            [-1],
            [1],
            [-1],
            [-1],
            [-1],
            [B_direction],
            [-1],
            [1],
            [-1],
        ],
        dtype=tf.float32,
    )

    # Flip node 8
    new_spin_state = tf.constant(
        [
            [1],
            [1],
            [1],
            [-1],
            [1],
            [-1],
            [-1],
            [-1],
            [-B_direction],
            [-1],
            [1],
            [-1],
        ],
        dtype=tf.float32,
    )

    old_log_proba = env.calculate_log_probas_of_spin_states(old_spin_state)
    new_log_proba = env.calculate_log_probas_of_spin_states(new_spin_state)
    delta_log_proba = new_log_proba - old_log_proba

    assert delta_log_proba < 0



#
# def test1():
#     params = tf.Variable([[2.0, 3.0], [1.0, 5.0]], dtype=tf.float32)
#     x = tf.constant([[2.0], [1.0]], dtype=tf.float32)
#     y = tf.constant([4.0, 2.0, 3.0, 1.0], dtype=tf.float32)
#
#     with tf.GradientTape() as tape:
#         u = tf.linalg.matmul(params, x)
#         v = tf.reshape(params, shape=(-1,))*y
#         w = tf.reduce_sum(u + tf.reduce_sum(v))
#     grad = tape.gradient(w, params)
#
#     expected_grad = tf.concat([tf.transpose(x), tf.transpose(x)], axis=0) + 2*tf.reshape(y, shape=(2, 2))
#
#     tf.debugging.assert_equal(grad, expected_grad)
#
#     res = tf.reduce_sum(expected_grad*params)
#
#     tf.debugging.assert_equal(w, res)
#     # param = [[a, b],
#     #          [c, d]]
#     # u = [[ax1 + bx2],
#     #      [cx1 + dx2]]
#     # v = [ay1, by2, cy3, dy4]
#     # w = ax1 + bx2 + cx1 + dx2 + 2(ay1 + by2 + cy3 + dy4)
#     #   = a(x1 + 2y1) + b(x2 + 2y2) + c(x1 + 2y3) + d(x2 + 2y4)
#
#
#
# def test2():
#     a = tf.constant([
#         [1, 10],
#         [2, 20],
#         [3, 30]
#     ], dtype=tf.float32)
#
#     b = tf.reshape(tf.transpose(a), shape=(-1, 1))
#     print(b)
#
#
# def test3():
#     n_nodes = 3
#     a = tf.constant([[1], [2], [3], [10], [20], [30]], dtype=tf.float32)
#
#     # print(tf.squeeze(a))
#     b = tf.transpose(tf.reshape(a, shape=(-1, n_nodes)))
#     print(b)