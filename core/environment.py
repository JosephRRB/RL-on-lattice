import copy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import dgl
import tensorflow as tf


class SpinEnvironment:
    def __init__(
        self,
        lattice,
        inverse_temp=1,
        spin_coupling=1,
        external_B=0,
    ):
        self.inverse_temp = inverse_temp
        self.spin_coupling = spin_coupling
        self.external_B = external_B

        self.lattice = lattice
        self.n_nodes = self.lattice.num_nodes()
        self.spin_state = None

    def reset(self):
        random_ints = tf.random.uniform(
            shape=(self.n_nodes, 1),
            maxval=2,
            dtype=tf.int32,
        )
        self.spin_state = tf.cast(
            2 * random_ints - 1,
            dtype=tf.float32,
        )
        return self.spin_state

    # @tf.function
    def step(self, selected_nodes):
        """
        selected_nodes: node indices selected by agent
        """
        old_spins = self.spin_state

        # flip spins
        node_idxs = tf.transpose(selected_nodes)
        ones = tf.ones_like(node_idxs, dtype=tf.float32)
        encoded_selection = tf.scatter_nd(
            node_idxs, ones, shape=(self.n_nodes, 1)
        )
        flip_action = 1 - 2 * encoded_selection
        new_spins = old_spins * flip_action

        # reward
        reward = _calculate_reward(old_spins, new_spins)
        clipped_reward = tf.clip_by_value(
            reward, clip_value_min=1e-12, clip_value_max=1
        )

        # new observation
        self.spin_state = new_spins
        return self.spin_state, clipped_reward

    def calculate_log_probas_of_spin_states(self, spin_states):
        reshaped_spins = tf.transpose(
            tf.reshape(spin_states, shape=(-1, self.n_nodes))
        )
        with self.lattice.local_scope():
            self.lattice.ndata["spin"] = reshaped_spins
            self.lattice.apply_edges(
                dgl.function.v_mul_u("spin", "spin", "spin_interaction")
            )
            # number of edges were doubled because undirected edges
            # are represented as two oppositely directed edges
            total_spin_interaction = (
                dgl.readout_edges(self.lattice, "spin_interaction", op="sum")
                / 2
            )
        total_spin = tf.reduce_sum(reshaped_spins, axis=0, keepdims=True)
        negative_energy = (
            self.spin_coupling * total_spin_interaction
            + self.external_B * total_spin
        )
        log_probability = self.inverse_temp * tf.transpose(negative_energy)
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


# Information Theoretic Measures for Clusterings Comparison:
# Variants, Properties, Normalization and Correction for Chance
# Nguyen Xuan Vinh, Julien Epps, James Bailey
# Comparing clusterings by the variation of information
# Marina Meil ̆a
# May need to change reward function
# @tf.function(experimental_relax_shapes=True)
@tf.function
def _calculate_reward(old_spins, new_spins):
    old_feats = tf.reshape((old_spins + 1) / 2, shape=(-1,))
    new_feats = tf.reshape((new_spins + 1) / 2, shape=(-1,))

    joint_counts = tf.math.confusion_matrix(
        old_feats, new_feats, num_classes=2, dtype=tf.float32
    )
    counts_old = tf.reduce_sum(joint_counts, axis=1, keepdims=True)
    counts_new = tf.reduce_sum(joint_counts, axis=0, keepdims=True)
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


# @tf.function(experimental_relax_shapes=True)
@tf.function
def _calculate_entropy(counts, total_counts):
    entropy = -tf.reduce_sum(
        tf.where(
            tf.not_equal(counts, 0),
            counts
            * (tf.math.log(counts) - tf.math.log(total_counts))
            / total_counts,
            0,
        ),
        keepdims=True,
    )
    return entropy
