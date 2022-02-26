import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

    initial_weights = runner.agent.policy_network.get_weights()
    _ = runner.run_trajectory(n_transitions=2)
    final_weights = runner.agent.policy_network.get_weights()

    for w0, w1 in zip(initial_weights, final_weights):
        tf.debugging.assert_equal(w0, w1)


def test_train_step_updates_weights_of_policy_network():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    runner = Runner(SpinEnvironment(lattice), RLAgent(lattice))

    initial_weights = runner.agent.policy_network.get_weights()

    n_transitions_per_training_step = 2
    runner.batched_graphs_for_training = _create_batched_graphs(
        runner.agent.graph, n_batch=n_transitions_per_training_step
    )
    runner._training_step(n_transitions=n_transitions_per_training_step)
    final_weights = runner.agent.policy_network.get_weights()

    assert any([
        any(tf.reshape(tf.math.not_equal(w0, w1), shape=(-1,)))
        for w0, w1 in zip(initial_weights, final_weights)
    ])


def test_evaluate_does_not_change_policy_network_weights():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    agent = RLAgent(lattice)

    runner = Runner(environment, agent)
    evaluate_for_n_transitions = 100
    runner.batched_graphs_for_evaluation = _create_batched_graphs(
        runner.agent.graph, n_batch=evaluate_for_n_transitions
    )

    _ = runner._evaluate(evaluate_for_n_transitions=evaluate_for_n_transitions)
    initial_weights = runner.agent.policy_network.get_weights()

    _ = runner._evaluate(evaluate_for_n_transitions=evaluate_for_n_transitions)
    final_weights = runner.agent.policy_network.get_weights()

    for w0, w1 in zip(initial_weights, final_weights):
        tf.debugging.assert_equal(w0, w1)


def test_example():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(
        lattice,
        inverse_temp=0.5,
        spin_coupling=0.5,
        external_B=0
    )
    agent = RLAgent(
        lattice,
        n_hidden=10,
        learning_rate=0.0005
    )

    runner = Runner(environment, agent)

    train_ave_rewards = runner.train(
        n_training_loops=10000, n_transitions_per_training_step=2,
        evaluate_after_n_training_steps=100, evaluate_for_n_transitions=50
    )

    np.savetxt("data/example_training_result.csv", train_ave_rewards.numpy(), delimiter=",")

    plt.plot(train_ave_rewards)
    plt.savefig("data/sample_training_plot.png")