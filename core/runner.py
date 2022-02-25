import copy

import dgl
import tensorflow as tf


class Runner:
    def __init__(self, environment, agent):
        # self.n_transitions = n_transitions

        self.environment = environment
        self.agent = agent

        # self.agent.create_batched_graphs(n_batch=self.n_transitions)
        self.environment.reset()

        self.batched_graphs_for_training = None
        self.batched_graphs_for_evaluation = None

    def run_trajectory(self, n_transitions=2):
        old_obs = []
        actions = []
        new_obs = []
        rewards = []

        observation = self.environment.spin_state
        for _ in range(n_transitions):
            old_obs.append(observation)
            action_index = self.agent.act(observation)
            actions.append(action_index)
            observation, reward = self.environment.step(action_index)
            new_obs.append(observation)
            rewards.append(reward)

        # # Store last state
        # self.current_state = observation

        old_obs = tf.concat(old_obs, axis=0)
        actions = tf.concat(actions, axis=0)
        new_obs = tf.concat(new_obs, axis=0)
        rewards = tf.concat(rewards, axis=0)

        return old_obs, actions, new_obs, rewards

    def _training_step(self, n_transitions=2):
        "Batch graphs before running _training_step"
        old_obs, actions, new_obs, rewards = self.run_trajectory(n_transitions=n_transitions)

        with tf.GradientTape() as tape:
            log_probas_for_loss, bidir_acceptance_log_probas = self._calculate_log_probas(
                self.batched_graphs_for_training, old_obs, actions, new_obs
            )

            grad_weights_logits = tf.stop_gradient(
                bidir_acceptance_log_probas + tf.math.log(rewards)
            )
            grad_weights = tf.nn.softmax(grad_weights_logits, axis=0)
            # theoretical ave of grad_weights
            baseline = 1 / n_transitions
            advantages = grad_weights - baseline

            # negative log prob -> gradient descent
            loss = tf.reduce_sum(-log_probas_for_loss * advantages)

        grads = tape.gradient(loss, self.agent.policy_network.trainable_weights)
        self.agent.optimizer.apply_gradients(zip(grads, self.agent.policy_network.trainable_weights))

    def _calculate_log_probas(self, graphs, old_obs, actions, new_obs):
        old_log_probas = self.environment.calculate_log_probas_of_spin_states(old_obs)
        new_log_probas = self.environment.calculate_log_probas_of_spin_states(new_obs)
        delta_log_probas = new_log_probas - old_log_probas

        forward_log_probas = self.agent.calculate_log_probas_of_agent_actions(graphs, old_obs, actions)
        # Same action would map back the state
        backward_log_probas = self.agent.calculate_log_probas_of_agent_actions(graphs, new_obs, actions)

        bidir_acceptance_log_probas = -tf.abs(
            delta_log_probas + backward_log_probas - forward_log_probas
        ) / 2
        log_probas_for_loss = bidir_acceptance_log_probas + (forward_log_probas + backward_log_probas) / 2
        return log_probas_for_loss, bidir_acceptance_log_probas

    def train(self, n_training_loops=2000, n_transitions_per_training_step=2,
              evaluate_after_n_training_steps=50, evaluate_for_n_transitions=100):
        self.batched_graphs_for_training = _create_batched_graphs(
            self.agent.graph, n_batch=n_transitions_per_training_step
        )
        self.batched_graphs_for_evaluation = _create_batched_graphs(
            self.agent.graph, n_batch=evaluate_for_n_transitions
        )
        expected_rewards = []
        for i in range(n_training_loops):
            self._training_step(n_transitions=n_transitions_per_training_step)
            if i % evaluate_after_n_training_steps == 0:
                expected_reward = self._evaluate(evaluate_for_n_transitions=evaluate_for_n_transitions)
                expected_rewards.append(expected_reward)

        expected_rewards = tf.concat(expected_rewards, axis=0)
        return expected_rewards

    def _evaluate(self, evaluate_for_n_transitions=100):
        "Batch graphs before running _evaluate"
        old_obs, actions, new_obs, rewards = self.run_trajectory(n_transitions=evaluate_for_n_transitions)
        _, bidir_acc_log_probas = self._calculate_log_probas(
            self.batched_graphs_for_evaluation, old_obs, actions, new_obs
        )
        expected_reward = tf.reduce_mean(tf.math.exp(bidir_acc_log_probas)*rewards)
        return expected_reward


def _create_batched_graphs(graph, n_batch=2):
    batch_graphs = dgl.batch([graph] * n_batch)
    return batch_graphs