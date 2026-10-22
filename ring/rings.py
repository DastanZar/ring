"""The collectives: ring allreduce / reduce-scatter / allgather / broadcast.

allreduce = reduce-scatter (phase 1) + allgather (phase 2); 2(N-1) steps,
one S/N chunk moved per rank per step. Wire bytes per rank: 2(N−1)/N · S —
the NCCL derivation, measurable on plain sockets here.

The index law (derived by contributor-set simulation, enforced by the
equivalence tests): at step s every rank sends buffer[(r − s) mod N] right
and merges what it receives at index (r − 1 − s) mod N.

    phase 1 (reduce-scatter): ADD. Rank r starts its own lane at
        send_start = r−1 (so receive #0 lands in buffer r); after N−1 steps
        buffer r is the full sum.                            (sim-verified)
    phase 2 (allgather):     ASSIGN, same rotation: each rank forwards the
        sums one hop right per step until every slot is filled.

Send/recv ORDER alternates per step (even r sends first, odd r receives
first): without it, two ranks can both block in send() against full socket
buffers — a deadlock NCCL avoids with multiple QPs; we avoid it with a
one-line parity rule, documented at the socket so nobody "simplifies" it.

Socket capacity: init() sets 16 MB buffers, so a chunk bigger than that can
stall the ring (the error mode the alternation exists to survive).
Serialization is pickle-per-chunk: the ALGORITHM is under test; the
transport is incidental.
"""
from __future__ import annotations

import pickle
from typing import List, Sequence

import numpy as np

from .bootstrap import World, world


def _split(x: np.ndarray, n: int) -> List[np.ndarray]:
    return [a.astype(np.float64, copy=True) for a in np.array_split(x.reshape(-1), n)]


def _join(parts: Sequence[np.ndarray], shape) -> np.ndarray:
    return np.concatenate(list(parts)).reshape(shape)


def _per_rank(inputs, N: int) -> List[np.ndarray]:
    if isinstance(inputs, list):
        assert len(inputs) == N, "list form must give one array per rank"
        return [np.asarray(a) for a in inputs]
    return [np.asarray(inputs) for _ in range(N)]


def _exchange(rank, buffers: List, N: int, add: bool, start: int) -> None:
    """One ring phase. At step s rank r sends buffer (start−s) and merges
    what arrives into buffer (start−1−s). Rotating DOWNWARD is load-bearing:
    a buffer is received/merged at step s and SENT onward at s+1, so partial
    sums chain — rotating upward (my first draft) sends buffers nobody has
    updated yet and reduces nothing (a simulation caught it).

    start = r−1 makes phase 1 finish chunk r's full sum in rank r's buffer
    (receiver of the final addition), and phase 2 with the same start then
    forwards each already-complete buffer exactly one step after catching it.
    """
    r = rank.rank
    for s in range(N - 1):
        out_c = (start - s) % N
        in_c = (start - 1 - s) % N
        if r % 2 == 0:                            # parity alternation: no two
            rank.send(pickle.dumps(buffers[out_c]))   # neighbors both block in
            got = pickle.loads(rank.recv())           # send() on full buffers
        else:
            got = pickle.loads(rank.recv())
            rank.send(pickle.dumps(buffers[out_c]))
        buffers[in_c] = (buffers[in_c] + got) if add else got


def allreduce(inputs, w: World | None = None) -> List[np.ndarray]:
    """Ring allreduce. `inputs`: ONE array (every rank contributes it) or a
    list of N per-rank arrays. Returns per-rank results — all equal to
    Σ inputs; that equality is the test."""
    w = w or world()
    N = w.size
    data = _per_rank(inputs, N)

    def step(rank):
        r = rank.rank
        bufs = _split(data[r], N)
        _exchange(rank, bufs, N, add=True, start=r - 1)     # phase 1: bufs[r] full
        _exchange(rank, bufs, N, add=False, start=r)        # phase 2: allgather law
        return _join(bufs, data[r].shape)

    return w.run(step)


def reduce_scatter(inputs, w: World | None = None) -> List[np.ndarray]:
    """Phase 1 only: rank r returns the summed chunk r."""
    w = w or world()
    N = w.size
    data = _per_rank(inputs, N)

    def step(rank):
        r = rank.rank
        bufs = _split(data[r], N)
        _exchange(rank, bufs, N, add=True, start=r - 1)
        return bufs[r]

    return w.run(step)


def allgather(parts, w: World | None = None) -> List[np.ndarray]:
    """Rank r contributes parts[r]; every rank returns the concatenation."""
    w = w or world()
    N = w.size

    def step(rank):
        r = rank.rank
        full: List = [None] * N                    # type: ignore[list-item]
        full[r] = np.asarray(parts[r])
        _exchange(rank, full, N, add=False, start=r)
        return np.concatenate(full)                # index order: parts[0]..parts[N-1]

    return w.run(step)


def broadcast(x, root: int = 0, w: World | None = None) -> List[np.ndarray]:
    """Root's tensor circulates the ring once; non-root ranks pass None."""
    w = w or world()
    N = w.size

    def step(rank):
        r = rank.rank
        if r == root:
            arr = np.asarray(x)
            rank.send(pickle.dumps(arr))
            return arr
        arr = pickle.loads(rank.recv())
        if r != (root - 1) % N:                     # the rank before root closes it
            rank.send(pickle.dumps(arr))
        return arr

    return w.run(step)


def busbw(size_bytes: float, seconds: float, N: int) -> dict:
    """NCCL-style report: algbw = S/t; busbw = algbw · 2(N−1)/N."""
    algbw = size_bytes / seconds
    return {
        "algbw_gbps": round(algbw / 1e9 * 8, 3),
        "busbw_gbps": round(algbw * 2 * (N - 1) / N / 1e9 * 8, 3),
        "wire_bytes_per_rank": round(2 * (N - 1) / N * size_bytes, 1),
    }
