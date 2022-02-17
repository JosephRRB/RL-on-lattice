import copy

import tensorflow as tf


class Runner:
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent
        self.lattice = environment.lattice

        self.current_state = environment.reset()

    def run(self, n_transitions=3):
        old_obs = []
        actions = []
        new_obs = []
        rewards = []
        env_dlps = []

        observation = self.current_state
        for _ in range(n_transitions):
            old_obs.append(observation)
            action_index = self.agent.act(observation)
            actions.append(action_index)
            observation, reward, env_dlp = self.environment.step(action_index)
            new_obs.append(observation)
            rewards.append(reward)
            env_dlps.append(env_dlp)

        # Store last state
        self.current_state = observation

        actions = tf.concat(actions, axis=0)
