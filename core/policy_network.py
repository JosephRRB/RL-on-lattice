import dgl
from dgl.nn.tensorflow.conv import SAGEConv
import tensorflow as tf

# from tensorflow.keras import Model
# from tensorflow.keras.activations import tanh


class GraphPolicyNetwork(tf.keras.Model):
    def __init__(self, n_hidden, n_nodes):
        super().__init__()
        self.n_nodes = n_nodes
        n_node_features = 1
        self.layer_1 = SAGEConv(
            n_node_features,
            n_hidden,
            aggregator_type="mean",
        )
        self.layer_1.fc_self.build(n_node_features)
        self.layer_1.fc_neigh.build(n_node_features)
        self.layer_2 = SAGEConv(
            n_hidden, n_node_features, aggregator_type="mean"
        )
        self.layer_2.fc_self.build(n_hidden)
        self.layer_2.fc_neigh.build(n_hidden)
        self.layer_3 = tf.keras.layers.Dense(self.n_nodes)
        self.layer_3.build(2*self.n_nodes)

    def call(self, graphs, node_features):
        hidden1 = self.layer_1(graphs, node_features)
        hidden1 = tf.keras.activations.tanh(hidden1)
        output1 = self.layer_2(graphs, hidden1)
        output1 = tf.reshape(output1, shape=(-1, self.n_nodes))

        node_features_ = tf.reshape(node_features, shape=(-1, self.n_nodes))
        hidden2 = tf.concat([output1, node_features_], axis=1)
        hidden2 = tf.keras.activations.tanh(hidden2)
        output2 = self.layer_3(hidden2)
        return output1, output2


def _create_batched_graphs(graph, n_batch=2):
    batch_graphs = dgl.batch([graph] * n_batch)
    return batch_graphs
