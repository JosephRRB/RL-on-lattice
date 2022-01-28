import copy

import dgl
import tensorflow as tf
from core.policy_network import GraphPolicyNetwork
from tensorflow.keras.optimizers import Adam


class BaseAgent:
    def __init__(
        self,
        environment,
    ):
        self.environment = environment

    def act(self, observation):
        pass

    @staticmethod
    def _encode_action(action_index):
        encoded_action = tf.one_hot(
            tf.reshape(action_index, shape=(-1,)), depth=2
        )
        return encoded_action

    def run_trajectories(self, run_n_trajectories=5):
        old_obs = []
        actions = []
        new_obs = []
        rewards = []

        observation = self.environment.reset()  # remove reset
        for _ in range(run_n_trajectories):
            old_obs.append(observation)
            action_index = self.act(observation)
            actions.append(action_index)
            observation, reward = self.environment.step(action_index)
            new_obs.append(observation)
            rewards.append(reward)

        batch_old_obs = dgl.batch(old_obs)
        batch_new_obs = dgl.batch(new_obs)
        batch_actions = self._encode_action(tf.concat(actions, axis=0))
        batch_rewards = tf.reshape(tf.concat(rewards, axis=0), shape=(-1, 1))

        return batch_old_obs, batch_new_obs, batch_actions, batch_rewards

    def _action_log_probability(
        self, observation, encoded_action, training=False
    ):
        pass

    def calculate_acceptance_probas(self, run_n_trajectories=5):
        batch_old_obs, batch_new_obs, batch_actions, _ = self.run_trajectories(
            run_n_trajectories=run_n_trajectories
        )
        delta_log_proba_of_states = (
            self.environment.calculate_delta_log_proba_of_state(
                batch_old_obs, batch_new_obs
            )
        )
        forward_log_probas = self._action_log_probability(
            batch_old_obs, batch_actions, training=False
        )
        # Same action would map back the state
        backward_log_probas = self._action_log_probability(
            batch_new_obs, batch_actions, training=True
        )
        acc_log_probas = tf.math.minimum(
            0,
            delta_log_proba_of_states
            + backward_log_probas
            - forward_log_probas,
        )
        acc_probas = tf.math.exp(acc_log_probas)
        return acc_probas


class RLAgent(BaseAgent):
    def __init__(self,
                 environment,
                 n_hidden=10,
                 learning_rate=0.0005
                 ):
        super().__init__(environment)
        self.policy_network = GraphPolicyNetwork(
            1, n_hidden, 2
        )
        self.optimizer = Adam(learning_rate=learning_rate)

    def act(self, observation):
        logits = self.policy_network(observation)
        action_index = tf.random.categorical(logits, 1)
        return action_index

    def _action_log_probability(self, observation, encoded_action,
                                training=False):
        logits = self.policy_network(observation, training=training)
        probas = tf.nn.softmax(logits)
        action_probas = tf.reduce_sum(
            tf.math.multiply(probas, encoded_action),
            axis=1, keepdims=True
        )
        with observation.local_scope():
            observation.ndata['log_action_probas'] = tf.math.log(action_probas)
            log_probas = dgl.readout_nodes(observation, 'log_action_probas')
        return log_probas

    def _learn_on_trajectories(self, run_n_trajectories=5):
        # generate batch of data
        batch_old_obs, batch_new_obs, batch_actions, batch_rewards = self.run_trajectories(
            run_n_trajectories=run_n_trajectories
        )

        # env delta log proba
        delta_log_proba_of_states = self.environment.calculate_delta_log_proba_of_state(
            batch_old_obs, batch_new_obs
        )

        with tf.GradientTape() as tape:
            forward_log_probas = self._action_log_probability(
                batch_old_obs, batch_actions, training=True
            )
            # Same action would map back the state
            backward_log_probas = self._action_log_probability(
                batch_new_obs, batch_actions, training=True
            )

            bidir_acc_log_probas = -tf.abs(
                delta_log_proba_of_states + backward_log_probas - forward_log_probas
            ) / 2
            log_probs = bidir_acc_log_probas + (
                        forward_log_probas + backward_log_probas) / 2
            grad_weights_logits = tf.stop_gradient(
                bidir_acc_log_probas + tf.math.log(batch_rewards)
            )
            baseline = 1 / grad_weights_logits.shape[0]
            advantages = tf.nn.softmax(grad_weights_logits, axis=0) - baseline

            # negative log prob -> gradient descent
            loss = tf.reduce_sum(-log_probs * advantages)

        grads = tape.gradient(loss, self.policy_network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.policy_network.trainable_weights))

        return loss

    def learn(self, run_n_trajectories=5, train_n_epochs=100):
        losses = []
        for i in range(train_n_epochs):
            loss = self._learn_on_trajectories(
                run_n_trajectories=run_n_trajectories)
            losses.append(loss)

            if i % 100 == 0:
                print(f'{i + 1}/{train_n_epochs} training epochs')
        losses = tf.concat(losses, axis=0)
        return losses


class RandomAgent(BaseAgent):
    def __init__(self,
                 environment,
                 ):
        super().__init__(environment)

    def act(self, observation):
        n_nodes = observation.num_nodes()

        # equal probs
        logits = tf.ones(shape=(n_nodes, 2), dtype=tf.float32)
        action_index = tf.random.categorical(logits, 1)
        return action_index

    def _action_log_probability(self, observation, encoded_action,
                                training=False):
        n_graphs = observation.batch_size
        num_nodes_per_graph = tf.cast(
            tf.reshape(observation.batch_num_nodes(), shape=(-1, 1)),
            dtype=tf.float32
        )

        # equal probs
        log_proba = -num_nodes_per_graph * tf.math.log(
            2 * tf.ones(shape=(n_graphs, 1), dtype=tf.float32)
        )
        return log_proba
