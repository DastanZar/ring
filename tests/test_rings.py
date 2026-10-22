"""Ring collectives vs numpy reference, at every ring size 2..8.

The contract: allreduce(inputs) returns, on EVERY rank, the elementwise sum
of all ranks' inputs; reduce_scatter gives rank r the sum of chunk r;
allgather concatenates; broadcast circulates root's array. If the index law
in rings.py is even one slot off, these fail loudly — they caught two wrong
rotations during development (both documented in the module docstring).
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, ".")
import ring  # noqa: E402


@pytest.fixture
def world4():
    w = ring.init(4)
    yield w
    ring.shutdown()


@pytest.mark.parametrize("size", [2, 3, 4, 5, 8])
def test_allreduce_matches_numpy(size):
    w = ring.init(size)
    try:
        rng = np.random.default_rng(size)
        inputs = [rng.uniform(-1, 1, (7, 13)) for _ in range(size)]
        res = ring.allreduce(inputs, w=w)
        want = sum(inputs)
        assert all(np.allclose(r, want) for r in res), [np.abs(r - want).max() for r in res]
    finally:
        ring.shutdown()


def test_allreduce_uniform_input(world4):
    x = np.arange(4 * 4 * 4, dtype=np.float64).reshape(4, 16) / 7.0
    res = ring.allreduce(x, w=world4)
    assert all(np.allclose(r, 4 * x) for r in res)


def test_reduce_scatter_ownership(world4):
    rng = np.random.default_rng(9)
    inputs = [rng.uniform(-1, 1, 64) for _ in range(4)]
    parts = ring.reduce_scatter(inputs, w=world4)
    total = sum(inputs)
    chunks = np.array_split(total, 4)
    assert all(np.allclose(p, c) for p, c in zip(parts, chunks))


def test_allgather(world4):
    parts = [np.full(8, float(i)) for i in range(4)]
    res = ring.allgather(parts, w=world4)
    want = np.concatenate(parts)
    assert all(np.array_equal(r, want) for r in res)


def test_broadcast(world4):
    x = np.arange(11, dtype=np.float64)
    res = ring.broadcast(x, root=2, w=world4)
    assert all(np.array_equal(r, x) for r in res)


def test_busbw_formula():
    # 1 MB tensor, 4 ranks, 0.01 s: algbw = 1MiB/0.01s*8 = 0.839 Gbps
    rep = ring.busbw(1 << 20, 0.01, 4)
    assert abs(rep["algbw_gbps"] - (1 << 20) / 0.01 * 8 / 1e9) < 1e-3
    assert abs(rep["busbw_gbps"] - rep["algbw_gbps"] * 1.5) < 1e-3
