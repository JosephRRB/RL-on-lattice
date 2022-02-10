import copy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import dgl
import tensorflow as tf


class KagomeLatticeEnv:
    def __init__(
        self,
        n_sq_cells=20,
        inverse_temp=1,
        spin_coupling=1,
        external_B=0,
    ):
        self.inverse_temp = inverse_temp
        self.spin_coupling = spin_coupling
        self.external_B = external_B

        edge_list = _create_edge_list(n_sq_cells)
        self.coord_to_int, self.int_to_coord = _create_coord_int_mappings(
            edge_list
        )
        self.lattice = _create_lattice(edge_list, self.coord_to_int)
        self.spin_state = None

    def reset(self):
        random_ints = tf.random.uniform(
            shape=(self.lattice.num_nodes(), 1),
            maxval=2,
            dtype=tf.int32,
        )
        self.spin_state = tf.cast(
            2 * random_ints - 1,
            dtype=tf.float32,
        )
        return self.spin_state

    def step(self, action_index):
        """

        action_index: Tensor of shape (N_nodes, 1)
            that contains 0 (-1 spin) and 1 (+1 spin)
        """
        old_spins = self.spin_state

        # flip spins
        action = tf.cast(2 * action_index - 1, dtype=tf.float32)
        new_spins = old_spins * action

        reward = _calculate_reward(old_spins, new_spins)

        # new observation
        self.spin_state = new_spins
        return self.spin_state, reward



    #
    # def _calculate_log_proba_of_state(self, graph):
    #     with graph.local_scope():
    #         graph.apply_edges(
    #             dgl.function.v_mul_u("spin", "spin", "spin_interaction")
    #         )
    #         ## number of edges were doubled because undirected edges
    #         ## are represented as two oppositely directed edges
    #         total_spin_interaction = (
    #             dgl.readout_edges(graph, "spin_interaction") / 2
    #         )
    #         total_spin = dgl.readout_nodes(graph, "spin")
    #         negative_energy = (
    #             self.spin_coupling * total_spin_interaction
    #             + self.external_B * total_spin
    #         )
    #         log_probability = self.inverse_temp * negative_energy
    #         return log_probability
    #
    # def calculate_delta_log_proba_of_state(self, old_state, new_state):
    #     log_proba_of_old_state = self._calculate_log_proba_of_state(old_state)
    #     log_proba_of_new_state = self._calculate_log_proba_of_state(new_state)
    #     delta_log_proba_of_state = (
    #         log_proba_of_new_state - log_proba_of_old_state
    #     )
    #     return delta_log_proba_of_state
    #
    #
    #
    # def render_sub_lattice(self, center_node, radius):
    #     plt.figure(figsize=(10, 5))
    #
    #     graph = self.lattice.to_networkx(node_attrs=["spin"])
    #     sub_G = nx.ego_graph(
    #         graph, self.coord_to_int[center_node], radius=radius
    #     )
    #     pos = {
    #         n: np.array(self.int_to_coord[n])
    #         - np.array([0.5, np.sqrt(3) / 2]) * self.int_to_coord[n][1]
    #         for n in sub_G.nodes
    #     }
    #     color_map = [
    #         "blue" if sub_G.nodes[n]["spin"] == 1 else "red"
    #         for n in sub_G.nodes
    #     ]
    #     nx.draw(sub_G, pos=pos, node_color=color_map)


def _create_edge_list(n_sq_cells):
    v_max = 2 * n_sq_cells
    h_edges = [
        ((r, c), (r, (c + 1) % v_max))
        for r in range(0, v_max, 2)
        for c in range(v_max)
    ]
    ud_edges = [
        ((d, d + 1 + s), (d + 1, (d + 2 + s) % v_max))
        for s in range(0, v_max, 2)
        for d in range(v_max - 1 - s)
    ]
    h_ud_edges = h_edges + ud_edges
    v_ld_edges = [((v1[1], v1[0]), (v2[1], v2[0])) for v1, v2 in h_ud_edges]
    edge_list = h_ud_edges + v_ld_edges
    return edge_list


def _create_coord_int_mappings(edge_list):
    nodes = set.union(*[set(edge) for edge in edge_list])
    sorted_nodes = sorted(nodes, key=lambda x: (x[0], x[1]))
    coord_to_int_mapping = {
        coord_n: int_n
        for coord_n, int_n in zip(sorted_nodes, range(len(sorted_nodes)))
    }
    int_to_coord_mapping = {v: k for k, v in coord_to_int_mapping.items()}
    return coord_to_int_mapping, int_to_coord_mapping


def _create_lattice(edge_list, coord_to_int_map):
    graph = nx.Graph()
    graph.add_edges_from(edge_list)
    nx.relabel_nodes(graph, coord_to_int_map, copy=False)
    lattice = dgl.from_networkx(graph).to("/device:GPU:0")
    return lattice

def _calculate_reward(old_spins, new_spins):
    old_feats = tf.reshape((old_spins + 1) / 2, shape=(-1,))
    new_feats = tf.reshape((new_spins + 1) / 2, shape=(-1,))

    cross_tab = tf.math.confusion_matrix(
        old_feats, new_feats, num_classes=2, dtype=tf.float32
    )

    counts_old = tf.reduce_sum(cross_tab, axis=1, keepdims=True)
    counts_new = tf.reduce_sum(cross_tab, axis=0, keepdims=True)
    total_counts = tf.reduce_sum(cross_tab)

    mutual_info = tf.reduce_sum(
        tf.where(
            tf.not_equal(cross_tab, 0),
            cross_tab
            * (
                tf.math.log(cross_tab)
                - tf.math.log(counts_old)
                - tf.math.log(counts_new)
                + tf.math.log(total_counts)
            )
            / total_counts,
            0,
        )
    )

    old_entropy = -tf.reduce_sum(
        tf.where(
            tf.not_equal(counts_old, 0),
            counts_old
            * (tf.math.log(counts_old) - tf.math.log(total_counts))
            / total_counts,
            0,
        )
    )

    new_entropy = -tf.reduce_sum(
        tf.where(
            tf.not_equal(counts_new, 0),
            counts_new
            * (tf.math.log(counts_new) - tf.math.log(total_counts))
            / total_counts,
            0,
        )
    )
    var_info = old_entropy + new_entropy - 2 * mutual_info
    return var_info + 1e-9  ## add a small constant bias