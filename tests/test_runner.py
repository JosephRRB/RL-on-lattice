import tensorflow as tf

from core.agent import RLAgent
from core.environment import SpinEnvironment
from core.lattice import KagomeLattice
from core.runner import Runner


def test_runner_gives_correct_state_transitions():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    agent = RLAgent(lattice)

    runner = Runner(environment, agent, n_transitions=2)

    first_state = runner.environment.spin_state
    old_obs, _, new_obs, _ = runner.run_trajectory()

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

    runner = Runner(environment, agent, n_transitions=2)

    old_obs, actions, new_obs, rewards = runner.run_trajectory()

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



def test():
    lattice = KagomeLattice(n_sq_cells=2).lattice
    environment = SpinEnvironment(lattice)
    agent = RLAgent(lattice)

    runner = Runner(environment, agent, n_transitions=2)
    runner._training_step()