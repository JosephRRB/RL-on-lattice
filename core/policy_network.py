from dgl.nn.tensorflow.conv import SAGEConv
from tensorflow.keras import Model
from tensorflow.keras.activations import relu


class GraphPolicyNetwork(Model):
    def __init__(self,
                 graph,
                 n_node_features,
                 n_hidden,
                 n_classes):
        super().__init__()
        self.graph = graph
        self.layer_1 = SAGEConv(n_node_features, n_hidden,
                                aggregator_type='mean', activation=relu)
        self.layer_2 = SAGEConv(n_hidden, n_classes,
                                aggregator_type='mean')

    def call(self, inputs, training=None, mask=None):
        hidden = self.layer_1(self.graph, inputs, training=training)
        output_logits = self.layer_2(self.graph, hidden, training=training)
        return output_logits
