import tensorflow as tf

from core.policy_network import _create_batched_graphs


class Runner:
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent

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
            selected_nodes = self.agent.act(observation)
            actions.append(tf.RaggedTensor.from_tensor(selected_nodes))
            observation, reward = self.environment.step(selected_nodes)
            new_obs.append(observation)
            rewards.append(reward)

        old_obs = tf.concat(old_obs, axis=0)
        actions = tf.concat(actions, axis=0)
        new_obs = tf.concat(new_obs, axis=0)
        rewards = tf.concat(rewards, axis=0)

        return old_obs, actions, new_obs, rewards

    def _training_step(self, n_transitions=2):
        "Batch graphs before running _training_step"
        old_obs, actions, new_obs, rewards = self.run_trajectory(
            n_transitions=n_transitions
        )

        with tf.GradientTape() as tape:
            (
                delta_log_probas,
                forward_log_probas,
                backward_log_probas,
            ) = self._calculate_log_probas(
                self.batched_graphs_for_training, old_obs, actions, new_obs
            )
            bidir_acceptance_log_probas = (
                -tf.abs(
                    delta_log_probas + backward_log_probas - forward_log_probas
                )
                / 2
            )
            log_probas_for_loss = (
                bidir_acceptance_log_probas
                + (forward_log_probas + backward_log_probas) / 2
            )

            grad_weights_logits = tf.stop_gradient(
                bidir_acceptance_log_probas + tf.math.log(rewards)
            )
            grad_weights = tf.nn.softmax(grad_weights_logits, axis=0)
            # # theoretical ave of grad_weights
            # baseline = 1 / n_transitions

            baseline = tf.reduce_mean(grad_weights, axis=0, keepdims=True)
            advantages = grad_weights - baseline

            # negative log prob -> gradient descent
            loss = tf.reduce_sum(-log_probas_for_loss * advantages)

        grads = tape.gradient(
            loss, self.agent.policy_network.trainable_weights
        )
        self.agent.optimizer.apply_gradients(
            zip(grads, self.agent.policy_network.trainable_weights)
        )

    def _calculate_log_probas(self, graphs, old_obs, actions, new_obs):
        old_log_probas = self.environment.calculate_log_probas_of_spin_states(
            old_obs
        )
        new_log_probas = self.environment.calculate_log_probas_of_spin_states(
            new_obs
        )
        delta_log_probas = new_log_probas - old_log_probas

        forward_log_probas = self.agent.calculate_log_probas_of_agent_actions(
            graphs, old_obs, actions
        )
        # Reversed action would map back the state
        reversed_actions = actions[:, ::-1]
        backward_log_probas = self.agent.calculate_log_probas_of_agent_actions(
            graphs, new_obs, reversed_actions
        )
        return delta_log_probas, forward_log_probas, backward_log_probas

    def train(
        self,
        n_training_loops=2000,
        n_transitions_per_training_step=2,
        evaluate_after_n_training_steps=50,
        evaluate_for_n_transitions=100,
    ):
        self.batched_graphs_for_training = _create_batched_graphs(
            self.agent.graph, n_batch=n_transitions_per_training_step
        )
        self.batched_graphs_for_evaluation = _create_batched_graphs(
            self.agent.graph, n_batch=evaluate_for_n_transitions
        )
        train_acc_rate = []
        for i in range(n_training_loops):
            self._training_step(n_transitions=n_transitions_per_training_step)
            if i % evaluate_after_n_training_steps == 0:
                # print(f"Training step #: {i+1}")
                acc_rate = self._evaluate(
                    evaluate_for_n_transitions=evaluate_for_n_transitions
                )
                train_acc_rate.append(acc_rate)

        train_acc_rate = tf.concat(train_acc_rate, axis=0)
        return train_acc_rate

    def _evaluate(self, evaluate_for_n_transitions=100):
        "Batch graphs before running _evaluate"
        old_obs, actions, new_obs, rewards = self.run_trajectory(
            n_transitions=evaluate_for_n_transitions
        )
        (
            delta_log_probas,
            forward_log_probas,
            backward_log_probas,
        ) = self._calculate_log_probas(
            self.batched_graphs_for_evaluation, old_obs, actions, new_obs
        )
        acceptance_lp = tf.math.minimum(
            0, delta_log_probas + backward_log_probas - forward_log_probas
        )
        acceptance_prob = tf.math.exp(acceptance_lp)
        rand = tf.random.uniform(shape=acceptance_prob.shape)
        n_accepted = tf.reduce_sum(
            tf.cast(rand < acceptance_prob, dtype=tf.float32),
            axis=0,
            keepdims=True,
        )
        acceptance_rate = n_accepted / acceptance_prob.shape[0]

        return acceptance_rate
        # ave_reward = tf.reduce_mean(
        #     tf.math.exp(bidir_acc_log_probas) * rewards
        # )
        # return ave_reward
