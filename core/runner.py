import tensorflow as tf
from core.policy_network import _create_batched_graphs


class Runner:
    def __init__(
        self,
        environment,
        agent,
        prob_ratio_clip=0.2,
        max_n_updates_per_training_step=5,
        max_dist=0.01,
    ):
        self.prob_ratio_clip = prob_ratio_clip
        self.max_n_updates_per_training_step = max_n_updates_per_training_step
        self.max_dist = max_dist
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
        lp_to_maximize_0, _, _, bidir_policy_lp_0 = self._calculate_log_probas(
            self.batched_graphs_for_training, old_obs, actions, new_obs
        )
        for i in range(self.max_n_updates_per_training_step):
            bidir_policy_lp = self._update_policy_weights(
                old_obs, actions, new_obs, rewards, lp_to_maximize_0
            )
            ave_dist = tf.abs(
                tf.reduce_mean(bidir_policy_lp_0 - bidir_policy_lp)
            )
            if ave_dist > self.max_dist:
                # Early stopping
                break

    def _update_policy_weights(
        self, old_obs, actions, new_obs, rewards, lp_to_maximize_0
    ):
        with tf.GradientTape() as tape:
            (
                lp_to_maximize,
                bidir_acceptance_lp,
                _,
                bidir_policy_lp,
            ) = self._calculate_log_probas(
                self.batched_graphs_for_training, old_obs, actions, new_obs
            )
            prob_ratio = tf.math.exp(lp_to_maximize - lp_to_maximize_0)
            clipped_prob_ratio = tf.clip_by_value(
                prob_ratio,
                clip_value_min=1 - self.prob_ratio_clip,
                clip_value_max=1 + self.prob_ratio_clip,
            )

            # Advantages
            grad_weights_logits = tf.stop_gradient(
                bidir_acceptance_lp + tf.math.log(rewards)
            )
            grad_weights = tf.nn.softmax(grad_weights_logits, axis=0)
            baseline = tf.reduce_mean(grad_weights, axis=0, keepdims=True)
            advantages = grad_weights - baseline

            weighted_prob_ratio = tf.math.minimum(
                prob_ratio * advantages, clipped_prob_ratio * advantages
            )

            # negative log prob -> gradient descent
            # loss = tf.reduce_mean(-lp_to_maximize * advantages)
            loss = tf.reduce_mean(-weighted_prob_ratio)

        grads = tape.gradient(
            loss, self.agent.policy_network.trainable_weights
        )
        self.agent.optimizer.apply_gradients(
            zip(grads, self.agent.policy_network.trainable_weights)
        )
        return bidir_policy_lp

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

        state_action_delta_lp = (
            delta_log_probas + backward_log_probas - forward_log_probas
        )
        fwd_accept_lp = tf.math.minimum(0, state_action_delta_lp)

        bidir_accept_lp = -tf.abs(state_action_delta_lp) / 2
        bidir_policy_lp = (forward_log_probas + backward_log_probas) / 2
        lp_to_maximize = bidir_accept_lp + bidir_policy_lp
        return lp_to_maximize, bidir_accept_lp, fwd_accept_lp, bidir_policy_lp

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
        train_eval_results = []
        for i in range(n_training_loops):
            self._training_step(n_transitions=n_transitions_per_training_step)
            if i % evaluate_after_n_training_steps == 0:
                eval_results = self._evaluate(
                    evaluate_for_n_transitions=evaluate_for_n_transitions
                )
                train_eval_results.append(eval_results)

        train_eval_results = tf.concat(train_eval_results, axis=0)
        return train_eval_results

    def _evaluate(self, evaluate_for_n_transitions=100):
        "Batch graphs before running _evaluate"
        old_obs, actions, new_obs, rewards = self.run_trajectory(
            n_transitions=evaluate_for_n_transitions
        )
        _, bidir_acceptance_lp, acceptance_lp, _ = self._calculate_log_probas(
            self.batched_graphs_for_evaluation, old_obs, actions, new_obs
        )
        acceptance_prob = tf.math.exp(acceptance_lp)
        rand = tf.random.uniform(shape=acceptance_prob.shape)
        n_accepted = tf.reduce_sum(
            tf.cast(rand < acceptance_prob, dtype=tf.float32),
            axis=0,
            keepdims=True,
        )
        acceptance_rate = n_accepted / acceptance_prob.shape[0]

        ave_reward = tf.reduce_mean(
            tf.math.exp(bidir_acceptance_lp) * rewards,
            axis=0,
            keepdims=True,
        )

        eval_results = tf.concat([acceptance_rate, ave_reward], axis=1)
        return eval_results
