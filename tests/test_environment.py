import numpy as np

import tensorflow as tf
from core.environment import (
    KagomeLatticeEnv,
    _create_coord_int_mappings,
    _create_edge_list
)


def test_create_correct_node_connectivity():
    n_sq_cells = 2
    edge_list = _create_edge_list(n_sq_cells)
    expected = [
        ((0, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 2), (0, 3)),
        ((0, 3), (0, 0)),
        ((2, 0), (2, 1)),
        ((2, 1), (2, 2)),
        ((2, 2), (2, 3)),
        ((2, 3), (2, 0)),
        ((0, 0), (1, 0)),
        ((1, 0), (2, 0)),
        ((2, 0), (3, 0)),
        ((3, 0), (0, 0)),
        ((0, 2), (1, 2)),
        ((1, 2), (2, 2)),
        ((2, 2), (3, 2)),
        ((3, 2), (0, 2)),
        ((0, 1), (1, 2)),
        ((1, 2), (2, 3)),
        ((2, 3), (3, 0)),
        ((0, 3), (1, 0)),
        ((1, 0), (2, 1)),
        ((2, 1), (3, 2)),
        ((3, 2), (0, 3)),
        ((3, 0), (0, 1)),
    ]

    assert set(edge_list) == set(expected)


def test_correct_coord_int_node_mappings():
    n_sq_cells = 2
    edge_list = _create_edge_list(n_sq_cells)
    coord_to_int_mapping, int_to_coord_mapping = _create_coord_int_mappings(
        edge_list
    )
    expected_coord_to_int = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (1, 0): 4,
        (1, 2): 5,
        (2, 0): 6,
        (2, 1): 7,
        (2, 2): 8,
        (2, 3): 9,
        (3, 0): 10,
        (3, 2): 11,
    }
    expected_int_to_coord = {
        0: (0, 0),
        1: (0, 1),
        2: (0, 2),
        3: (0, 3),
        4: (1, 0),
        5: (1, 2),
        6: (2, 0),
        7: (2, 1),
        8: (2, 2),
        9: (2, 3),
        10: (3, 0),
        11: (3, 2),
    }
    assert coord_to_int_mapping == expected_coord_to_int
    assert int_to_coord_mapping == expected_int_to_coord


def test_lattice_has_correct_connectivity():
    environment = KagomeLatticeEnv(n_sq_cells=2)
    adjacency_of_lattice = environment.lattice.adj(scipy_fmt="coo").todense()

    expected = np.matrix(
        [
            # 0  1  2  3  4  5  6  7  8  9  10 11
            [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],  # 0
            [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # 1
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],  # 2
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # 3
            [1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],  # 4
            [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # 5
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],  # 6
            [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],  # 7
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],  # 8
            [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],  # 9
            [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # 10
            [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0],  # 11
        ]
    )

    np.testing.assert_array_equal(adjacency_of_lattice, expected)


def test_environment_gives_correct_spins():
    environment = KagomeLatticeEnv(n_sq_cells=2)
    observation = environment.reset()

    assert all(np.unique(observation) == [-1, 1])


def test_environment_resets():
    environment = KagomeLatticeEnv(n_sq_cells=20)
    observation_1 = environment.reset()
    observation_2 = environment.reset()

    assert any(observation_1 != observation_2)


def test_environment_correctly_tracks_lattice_spins_between_resets():
    environment = KagomeLatticeEnv(n_sq_cells=20)
    observation_1 = environment.reset()
    assert all(observation_1 == environment.spin_state)

    observation_2 = environment.reset()
    assert all(observation_2 == environment.spin_state)
