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
            that contains 1 (flip spin) and 0 (retain spin)
        """
        old_spins = self.spin_state

        # flip spins
        action = tf.cast(1 - 2 * action_index, dtype=tf.float32)
        new_spins = old_spins * action

        # reward
        reward = _calculate_reward(old_spins, new_spins)
        clipped_reward = tf.clip_by_value(
            reward, clip_value_min=1e-12, clip_value_max=1
        )

        # delta log proba of state transition
        log_proba_old = self._calculate_log_proba_of_spin_state(old_spins)
        log_proba_new = self._calculate_log_proba_of_spin_state(new_spins)
        delta_log_proba = log_proba_new - log_proba_old

        # new observation
        self.spin_state = new_spins
        return self.spin_state, clipped_reward, delta_log_proba

    def _calculate_log_proba_of_spin_state(self, spin_state):
        with self.lattice.local_scope():
            self.lattice.ndata["spin"] = spin_state
            self.lattice.apply_edges(
                dgl.function.v_mul_u("spin", "spin", "spin_interaction")
            )
            # number of edges were doubled because undirected edges
            # are represented as two oppositely directed edges
            total_spin_interaction = (
                dgl.readout_edges(self.lattice, "spin_interaction") / 2
            )
            total_spin = dgl.readout_nodes(self.lattice, "spin")
            negative_energy = (
                self.spin_coupling * total_spin_interaction
                + self.external_B * total_spin
            )
            log_probability = self.inverse_temp * negative_energy
            return log_probability

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


# Information Theoretic Measures for Clusterings Comparison:
# Variants, Properties, Normalization and Correction for Chance
# Nguyen Xuan Vinh, Julien Epps, James Bailey
# Comparing clusterings by the variation of information
# Marina Meil ̆a
@tf.function(experimental_relax_shapes=True)
def _calculate_reward(old_spins, new_spins):
    old_feats = tf.reshape((old_spins + 1) / 2, shape=(-1,))
    new_feats = tf.reshape((new_spins + 1) / 2, shape=(-1,))

    joint_counts = tf.math.confusion_matrix(
        old_feats, new_feats, num_classes=2, dtype=tf.float32
    )
    counts_old = tf.reduce_sum(joint_counts, axis=1)
    counts_new = tf.reduce_sum(joint_counts, axis=0)
    total_counts = tf.reduce_sum(joint_counts)

    joint_entropy = _calculate_entropy(joint_counts, total_counts)
    old_entropy = _calculate_entropy(counts_old, total_counts)
    new_entropy = _calculate_entropy(counts_new, total_counts)

    variation_of_info = 2 * joint_entropy - old_entropy - new_entropy
    normalized_vi = tf.where(
        tf.not_equal(joint_entropy, 0),
        variation_of_info / joint_entropy,
        0,
    )
    return normalized_vi


@tf.function(experimental_relax_shapes=True)
def _calculate_entropy(counts, total_counts):
    entropy = -tf.reduce_sum(
        tf.where(
            tf.not_equal(counts, 0),
            counts
            * (tf.math.log(counts) - tf.math.log(total_counts))
            / total_counts,
            0,
        )
    )
    return entropy
