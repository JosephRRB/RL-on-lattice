import copy

import tensorflow as tf


class Runner:
    def __init__(self, environment, agent, n_transitions=2):
        self.n_transitions = n_transitions

        self.environment = environment
        self.agent = agent

        self.agent.create_batched_graphs(n_batch=self.n_transitions)
        self.environment.reset()

    def run_trajectory(self):
        old_obs = []
        actions = []
        new_obs = []
        rewards = []

        observation = self.environment.spin_state
        for _ in range(self.n_transitions):
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

    def _training_step(self):
        old_obs, actions, new_obs, rewards = self.run_trajectory()

        old_log_probas = self.environment.calculate_log_probas_of_spin_states(old_obs)
        new_log_probas = self.environment.calculate_log_probas_of_spin_states(new_obs)
        delta_log_probas = new_log_probas - old_log_probas

        with tf.GradientTape(persistent=True) as tape:
            forward_log_probas = self.agent.calculate_log_probas_of_agent_actions(old_obs, actions)
            # Same action would map back the state
            backward_log_probas = self.agent.calculate_log_probas_of_agent_actions(new_obs, actions)

            bidir_acceptance_log_probas = -tf.abs(
                delta_log_probas + backward_log_probas - forward_log_probas
            ) / 2
            log_probs = bidir_acceptance_log_probas + (forward_log_probas + backward_log_probas) / 2
            grad_weights_logits = tf.stop_gradient(
                bidir_acceptance_log_probas + tf.math.log(rewards)
            )
            grad_weights = tf.nn.softmax(grad_weights_logits, axis=0)
            # theoretical ave of grad_weights
            baseline = 1 / self.n_transitions
            advantages = grad_weights - baseline

            # negative log prob -> gradient descent
            loss = tf.reduce_sum(-log_probs * advantages)

        grads = tape.gradient(loss, self.agent.policy_network.trainable_weights)
        self.agent.optimizer.apply_gradients(zip(grads, self.agent.policy_network.trainable_weights))


    # def train(self, n_training_loops=100):
    #
    #     for i in range(n_training_loops):




