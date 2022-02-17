from dgl.nn.tensorflow.conv import SAGEConv
from tensorflow.keras import Model
from tensorflow.keras.activations import relu


class GraphPolicyNetwork(Model):
    def __init__(self,
                 n_node_features,
                 n_hidden,
                 n_classes):
        super().__init__()
        self.layer_1 = SAGEConv(n_node_features, n_hidden,
                                aggregator_type='mean', activation=relu)
        self.layer_2 = SAGEConv(n_hidden, n_classes,
                                aggregator_type='mean')

    def call(self, graphs, node_features):
        hidden = self.layer_1(graphs, node_features)
        output_logits = self.layer_2(graphs, hidden)
        return output_logits
