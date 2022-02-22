import networkx as nx

import dgl


class KagomeLattice:
    def __init__(
        self,
        n_sq_cells=20,
    ):
        edge_list = _create_edge_list(n_sq_cells)
        self.coord_to_int, self.int_to_coord = _create_coord_int_mappings(
            edge_list
        )
        self.lattice = _create_lattice(edge_list, self.coord_to_int)


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
    lattice = dgl.from_networkx(graph)#.to("/device:GPU:0")
    return lattice
