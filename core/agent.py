import tensorflow as tf
from core.policy_network import GraphPolicyNetwork


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


# https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
# https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
# https://github.com/tensorflow/tensorflow/issues/9260
# http://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
# https://arxiv.org/abs/1903.06059
# https://arxiv.org/abs/1901.10517
@tf.function
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
    clipped = tf.clip_by_value(renorm_p, clip_value_min=2e-7, clip_value_max=1)
    renorm_lp = tf.math.log(clipped)

    action_log_probas = (
        tf.reduce_sum(selected_lp, axis=1, keepdims=True)
        - tf.reduce_sum(renorm_lp, axis=1, keepdims=True)
        + selected_n_lp
    )
    return action_log_probas
