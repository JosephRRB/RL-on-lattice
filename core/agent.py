import copy

import dgl
import tensorflow as tf
from core.policy_network import GraphPolicyNetwork

# from tensorflow.keras.optimizers import Adam


class RLAgent:
    def __init__(self, graph, n_hidden=10, learning_rate=0.0005):
        self.graph = graph
        self.graph_n_nodes = self.graph.num_nodes()
        self.policy_network = GraphPolicyNetwork(
            n_hidden=n_hidden, n_nodes=self.graph_n_nodes
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def act(self, observation):
        node_logits, n_nodes_logits = self.policy_network(
            self.graph, observation
        )

        # How many nodes to select
        select_k_nodes = int(tf.random.categorical(n_nodes_logits, 1)) + 1

        # Selected nodes in order
        selected_nodes = _choose_without_replacement(
            node_logits, select_k_nodes
        )
        return selected_nodes

    def calculate_log_probas_of_agent_actions(
        self, graphs, observations, selected_nodes
    ):
        "Receives batched graphs with corresponding observations and selected_nodes"
        node_logits, n_nodes_logits = self.policy_network(graphs, observations)

        log_proba = _calculate_action_log_probas_from_logits(
            node_logits, n_nodes_logits, selected_nodes
        )
        return log_proba




#
# class BaseAgent:
#     def __init__(
#         self,
#         environment,
#     ):
#         self.environment = environment
#
#     def act(self, observation):
#         pass
#
#     @staticmethod
#     def _encode_action(action_index):
#         encoded_action = tf.one_hot(
#             tf.reshape(action_index, shape=(-1,)), depth=2
#         )
#         return encoded_action
#
#     def run_trajectories(self, run_n_trajectories=5):
#         old_obs = []
#         actions = []
#         new_obs = []
#         rewards = []
#
#         observation = self.environment.reset()  # remove reset
#         for _ in range(run_n_trajectories):
#             old_obs.append(observation)
#             action_index = self.act(observation)
#             actions.append(action_index)
#             observation, reward = self.environment.step(action_index)
#             new_obs.append(observation)
#             rewards.append(reward)
#
#         batch_old_obs = dgl.batch(old_obs)
#         batch_new_obs = dgl.batch(new_obs)
#         batch_actions = self._encode_action(tf.concat(actions, axis=0))
#         batch_rewards = tf.reshape(tf.concat(rewards, axis=0), shape=(-1, 1))
#
#         return batch_old_obs, batch_new_obs, batch_actions, batch_rewards
#
#     def _action_log_probability(
#         self, observation, encoded_action, training=False
#     ):
#         pass
#
#     def calculate_acceptance_probas(self, run_n_trajectories=5):
#         batch_old_obs, batch_new_obs, batch_actions, _ = self.run_trajectories(
#             run_n_trajectories=run_n_trajectories
#         )
#         delta_log_proba_of_states = (
#             self.environment.calculate_delta_log_proba_of_state(
#                 batch_old_obs, batch_new_obs
#             )
#         )
#         forward_log_probas = self._action_log_probability(
#             batch_old_obs, batch_actions, training=False
#         )
#         # Same action would map back the state
#         backward_log_probas = self._action_log_probability(
#             batch_new_obs, batch_actions, training=True
#         )
#         acc_log_probas = tf.math.minimum(
#             0,
#             delta_log_proba_of_states
#             + backward_log_probas
#             - forward_log_probas,
#         )
#         acc_probas = tf.math.exp(acc_log_probas)
#         return acc_probas
#
#
# class RLAgent(BaseAgent):
#     def __init__(self,
#                  environment,
#                  n_hidden=10,
#                  learning_rate=0.0005
#                  ):
#         super().__init__(environment)
#         self.policy_network = GraphPolicyNetwork(
#             1, n_hidden, 2
#         )
#         self.optimizer = Adam(learning_rate=learning_rate)
#
#     def act(self, observation):
#         logits = self.policy_network(observation)
#         action_index = tf.random.categorical(logits, 1)
#         return action_index
#
#     def _action_log_probability(self, observation, encoded_action,
#                                 training=False):
#         logits = self.policy_network(observation, training=training)
#         probas = tf.nn.softmax(logits)
#         action_probas = tf.reduce_sum(
#             tf.math.multiply(probas, encoded_action),
#             axis=1, keepdims=True
#         )
#         with observation.local_scope():
#             observation.ndata['log_action_probas'] = tf.math.log(action_probas)
#             log_probas = dgl.readout_nodes(observation, 'log_action_probas')
#         return log_probas
#
#     def _learn_on_trajectories(self, run_n_trajectories=5):
#         # generate batch of data
#         batch_old_obs, batch_new_obs, batch_actions, batch_rewards = self.run_trajectories(
#             run_n_trajectories=run_n_trajectories
#         )
#
#         # env delta log proba
#         delta_log_proba_of_states = self.environment.calculate_delta_log_proba_of_state(
#             batch_old_obs, batch_new_obs
#         )
#
#         with tf.GradientTape() as tape:
#             forward_log_probas = self._action_log_probability(
#                 batch_old_obs, batch_actions, training=True
#             )
#             # Same action would map back the state
#             backward_log_probas = self._action_log_probability(
#                 batch_new_obs, batch_actions, training=True
#             )
#
#             bidir_acc_log_probas = -tf.abs(
#                 delta_log_proba_of_states + backward_log_probas - forward_log_probas
#             ) / 2
#             log_probs = bidir_acc_log_probas + (
#                         forward_log_probas + backward_log_probas) / 2
#             grad_weights_logits = tf.stop_gradient(
#                 bidir_acc_log_probas + tf.math.log(batch_rewards)
#             )
#             baseline = 1 / grad_weights_logits.shape[0]
#             advantages = tf.nn.softmax(grad_weights_logits, axis=0) - baseline
#
#             # negative log prob -> gradient descent
#             loss = tf.reduce_sum(-log_probs * advantages)
#
#         grads = tape.gradient(loss, self.policy_network.trainable_weights)
#         self.optimizer.apply_gradients(
#             zip(grads, self.policy_network.trainable_weights))
#
#         return loss
#
#     def learn(self, run_n_trajectories=5, train_n_epochs=100):
#         losses = []
#         for i in range(train_n_epochs):
#             loss = self._learn_on_trajectories(
#                 run_n_trajectories=run_n_trajectories)
#             losses.append(loss)
#
#             if i % 100 == 0:
#                 print(f'{i + 1}/{train_n_epochs} training epochs')
#         losses = tf.concat(losses, axis=0)
#         return losses
#
#
# class RandomAgent(BaseAgent):
#     def __init__(self,
#                  environment,
#                  ):
#         super().__init__(environment)
#
#     def act(self, observation):
#         n_nodes = observation.num_nodes()
#
#         # equal probs
#         logits = tf.ones(shape=(n_nodes, 2), dtype=tf.float32)
#         action_index = tf.random.categorical(logits, 1)
#         return action_index
#
#     def _action_log_probability(self, observation, encoded_action,
#                                 training=False):
#         n_graphs = observation.batch_size
#         num_nodes_per_graph = tf.cast(
#             tf.reshape(observation.batch_num_nodes(), shape=(-1, 1)),
#             dtype=tf.float32
#         )
#
#         # equal probs
#         log_proba = -num_nodes_per_graph * tf.math.log(
#             2 * tf.ones(shape=(n_graphs, 1), dtype=tf.float32)
#         )
#         return log_proba

#
# def _encode_action(action_index):
#     encoded_action = tf.one_hot(tf.reshape(action_index, shape=(-1,)), depth=2)
#     return encoded_action
#
#
# def _batch_calculate_log_proba_of_actions(logits, action_indices, n_batch):
#     log_probas = tf.nn.log_softmax(logits)
#     encoded_action = _encode_action(action_indices)
#     action_index_log_probas = tf.reduce_sum(
#         tf.math.multiply(log_probas, encoded_action), axis=1
#     )
#     batched_log_probas = tf.reshape(
#         action_index_log_probas, shape=(n_batch, -1)
#     )
#     log_proba_of_actions = tf.reduce_sum(
#         batched_log_probas, axis=1, keepdims=True
#     )
#     return log_proba_of_actions

# https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
# https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
# https://github.com/tensorflow/tensorflow/issues/9260
# http://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
# https://arxiv.org/abs/1903.06059
# https://arxiv.org/abs/1901.10517
def _choose_without_replacement(node_logits, select_k):
    uniform = tf.random.uniform(shape=node_logits.shape, minval=0, maxval=1)
    gumbel = -tf.math.log(-tf.math.log(uniform))
    _, node_indices = tf.nn.top_k(node_logits + gumbel, select_k)
    return node_indices

# TODO: check how to use tf.function here
# @tf.function(experimental_relax_shapes=True)
def _calculate_action_log_probas_from_logits(
    node_logits, n_nodes_logits, selected_nodes
):
    n_nodes = tf.reshape(selected_nodes.row_lengths(), shape=(-1, 1))
    n_nodes_lp = tf.nn.log_softmax(n_nodes_logits)
    selected_n_lp = tf.gather(
        params=n_nodes_lp, indices=n_nodes - 1, axis=1, batch_dims=1
    )

    node_lp = tf.nn.log_softmax(node_logits)
    selected_lp = tf.gather(
        params=node_lp, indices=selected_nodes, axis=1, batch_dims=1
    )
    selected_p_renorm = tf.math.exp(selected_lp[:, :-1])
    renorm_p = 1 - tf.map_fn(tf.cumsum, selected_p_renorm)
    clipped = tf.clip_by_value(
        renorm_p, clip_value_min=2e-7, clip_value_max=1
    )
    renorm_lp = tf.math.log(clipped)

    action_log_probas = (
        tf.reduce_sum(selected_lp, axis=1, keepdims=True)
        - tf.reduce_sum(renorm_lp, axis=1, keepdims=True)
        + selected_n_lp
    )
    return action_log_probas