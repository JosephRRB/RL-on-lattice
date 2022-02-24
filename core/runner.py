import copy

import tensorflow as tf


class Runner:
    def __init__(self, environment, agent, n_transitions=2):
        self.n_transitions = n_transitions

        self.environment = environment
        self.agent = agent

        self.agent._batch_graphs(n_batch=self.n_transitions)
        self.current_state = environment.reset()

    def run_trajectory(self):
        old_obs = []
        actions = []
        new_obs = []
        rewards = []

        observation = self.current_state
        for _ in range(self.n_transitions):
            old_obs.append(observation)
            action_index = self.agent.act(observation)
            actions.append(action_index)
            observation, reward = self.environment.step(action_index)
            new_obs.append(observation)
            rewards.append(reward)

        # Store last state
        self.current_state = observation

        old_obs = tf.concat(old_obs, axis=0)
        actions = tf.concat(actions, axis=0)
        new_obs = tf.concat(new_obs, axis=0)
        rewards = tf.concat(rewards, axis=0)

        return old_obs, actions, new_obs, rewards

    def _train_on_one_trajectory(self):
        old_obs, actions, new_obs, rewards = self.run_trajectory()




