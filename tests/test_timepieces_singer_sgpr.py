"""Tests for the explicitly limited single-tree SGPR analogues."""

import numpy as np
import pytest

from watchgen.mini_singer import (
    SimpleTree, select_cut, sgpr_acceptance_ratio,
    simulate_tree_height_variability, spr_move,
)


@pytest.fixture
def tree():
    return SimpleTree(
        {0: 4, 1: 4, 2: 5, 3: 5, 4: 6, 5: 6, 6: None},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0.3, 5: 0.8, 6: 1.5},
    )


def test_spr_preserves_binary_tree_and_positive_edges(tree):
    moved = spr_move(tree, cut_node=0, new_parent=2, new_time=0.5)
    assert len(moved.time) == len(tree.time)
    assert len(moved.branches()) == len(tree.branches())
    assert all(length > 0 for _, _, length in moved.branches())
    assert sorted(len(children) for children in moved.children.values()) == [2, 2, 2]


def test_spr_does_not_mutate_input(tree):
    original = dict(tree.parent)
    spr_move(tree, cut_node=0, new_parent=2, new_time=0.5)
    assert tree.parent == original


@pytest.mark.parametrize("kwargs", [
    {"cut_node": 0, "new_parent": 2, "new_time": 0},
    {"cut_node": 4, "new_parent": 0, "new_time": 0.1},
    {"cut_node": 6, "new_parent": 0, "new_time": 0.1},
])
def test_invalid_spr_is_rejected(tree, kwargs):
    with pytest.raises(ValueError):
        spr_move(tree, **kwargs)


def test_cut_lies_on_returned_branch(tree):
    node, time = select_cut(tree, np.random.default_rng(7))
    parent = tree.parent[node]
    assert tree.time[node] <= time < tree.time[parent]


def test_sgpr_height_ratio():
    assert sgpr_acceptance_ratio(2, 1.5) == 1
    assert sgpr_acceptance_ratio(1.5, 2) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        sgpr_acceptance_ratio(0, 1)


def test_coalescent_height_mean_is_known_value():
    n = 20
    heights = simulate_tree_height_variability(
        n, 50_000, np.random.default_rng(123))
    assert heights.mean() == pytest.approx(2 * (1 - 1 / n), rel=0.015)
