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

    first_state = runner.current_state
    old_obs, _, new_obs, _, _ = runner.run()

    old1, old2 = tf.split(old_obs, 2)
    new1, new2 = tf.split(new_obs, 2)

    last_state = runner.current_state

    tf.debugging.assert_equal(old1, first_state)
    tf.debugging.assert_equal(new1, old2)
    tf.debugging.assert_equal(new2, last_state)