import tensorflow as tf

from core.agent import RLAgent
from core.environment import SpinEnvironment
from core.lattice import KagomeLattice
from core.runner import Runner, _create_batched_graphs


def test_runner_gives_correct_state_transitions():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    agent = RLAgent(lattice)

    runner = Runner(environment, agent)

    first_state = runner.environment.spin_state
    old_obs, _, new_obs, _ = runner.run_trajectory(n_transitions=2)

    old1, old2 = tf.split(old_obs, 2)
    new1, new2 = tf.split(new_obs, 2)

    last_state = runner.environment.spin_state

    tf.debugging.assert_equal(old1, first_state)
    tf.debugging.assert_equal(new1, old2)
    tf.debugging.assert_equal(new2, last_state)


def test_actions_from_runner_are_consistent_with_environment_transitions():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    agent = RLAgent(lattice)

    runner = Runner(environment, agent)

    old_obs, actions, new_obs, rewards = runner.run_trajectory(n_transitions=2)

    old1, old2 = tf.split(old_obs, 2)
    act1, act2 = tf.split(actions, 2)
    new1, new2 = tf.split(new_obs, 2)
    r1, r2 = tf.split(rewards, 2)

    # -----------------------------------------------------------------------------------------------------------------
    environment.spin_state = old1
    expected_new1, expected_r1 = environment.step(act1)

    tf.debugging.assert_equal(new1, expected_new1)
    tf.debugging.assert_equal(r1, expected_r1)

    # -----------------------------------------------------------------------------------------------------------------

    environment.spin_state = old2
    expected_new2, expected_r2 = environment.step(act2)

    tf.debugging.assert_equal(new2, expected_new2)
    tf.debugging.assert_equal(r2, expected_r2)


def test_run_trajectory_does_not_change_weights_of_policy_network():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    runner = Runner(SpinEnvironment(lattice), RLAgent(lattice))

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

    # For a fixed input, calculate_log_probas_of_agent_actions only returns a different tensor
    # when the weights of the policy network are changed
    initial_res = runner.agent.calculate_log_probas_of_agent_actions(lattice, spin_state, agent_action_index)
    _ = runner.run_trajectory(n_transitions=2)
    next_res = runner.agent.calculate_log_probas_of_agent_actions(lattice, spin_state, agent_action_index)

    assert next_res == initial_res


def test_train_step_updates_weights_of_policy_network():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    runner = Runner(SpinEnvironment(lattice), RLAgent(lattice))

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

    # For a fixed input, calculate_log_probas_of_agent_actions only returns a different tensor
    # when the weights of the policy network are changed
    initial_res = runner.agent.calculate_log_probas_of_agent_actions(lattice, spin_state, agent_action_index)
    n_transitions_per_training_step = 2
    runner.batched_graphs_for_training = _create_batched_graphs(
        runner.agent.graph, n_batch=n_transitions_per_training_step
    )
    runner._training_step(n_transitions=n_transitions_per_training_step)
    next_res = runner.agent.calculate_log_probas_of_agent_actions(lattice, spin_state, agent_action_index)

    assert next_res != initial_res
    # assert tf.math.not_equal(next_res, initial_res)


def test():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    agent = RLAgent(lattice)

    runner = Runner(environment, agent)

    # runner.batched_graphs_for_training = _create_batched_graphs(
    #     runner.agent.graph, n_batch=2
    # )
    # runner._training_step(n_transitions=2)

    runner.batched_graphs_for_evaluation = _create_batched_graphs(
        runner.agent.graph, n_batch=100
    )
    expected_r1 = runner._evaluate(evaluate_for_n_transitions=100)

    expected_r2 = runner._evaluate(evaluate_for_n_transitions=100)

    expected_r2