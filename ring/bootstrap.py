"""Bootstrap: spawn ranks as threads on a TCP ring.

Each rank r has one outbound socket to (r+1) % N and one inbound from
(r-1) % N — a single logical ring, one NVLink-position equivalent.
Collectives run as a function executed concurrently by every rank thread;
blocking send/recv pairs the steps so no extra synchronization exists
(the ring flow control IS the barrier).
"""
from __future__ import annotations

import socket
import struct
import threading
from typing import Callable, List, Optional

_HEADER = struct.Struct("<Q")
_BUF = 1 << 20


def _send(sock: socket.socket, buf: bytes) -> None:
    sock.sendall(_HEADER.pack(len(buf)) + buf)


def _recv(sock: socket.socket) -> bytes:
    (n,) = _HEADER.unpack(_read_exactly(sock, 8))
    return _read_exactly(sock, n)


def _read_exactly(sock: socket.socket, n: int) -> bytes:
    chunks = []
    while n:
        c = sock.recv(min(n, _BUF))
        if not c:
            raise ConnectionError("ring link closed")
        chunks.append(c)
        n -= len(c)
    return b"".join(chunks)


class RankCtx:
    def __init__(self, rank: int, size: int, out: socket.socket, inp: socket.socket):
        self.rank, self.size = rank, size
        self._out, self._inp = out, inp

    def send(self, data: bytes) -> None:
        _send(self._out, data)

    def recv(self) -> bytes:
        return _recv(self._inp)


class World:
    def __init__(self, size: int):
        self.size = size
        self.ranks: List[RankCtx] = []
        self._threads: List[threading.Thread] = []
        self._socks: List[socket.socket] = []

    def run(self, fn: Callable[[RankCtx], object]) -> List:
        """Execute fn(rank) concurrently on every rank; return per-rank results."""
        results: List = [None] * self.size
        errors: List = [None] * self.size

        def body(r: int):
            try:
                results[r] = fn(self.ranks[r])
            except BaseException as e:  # noqa: BLE001 - re-raise on join
                errors[r] = e

        self._threads = [threading.Thread(target=body, args=(i,), daemon=True)
                         for i in range(self.size)]
        for t in self._threads:
            t.start()
        for t in self._threads:
            t.join(timeout=60)
        for t in self._threads:
            if t.is_alive():
                raise TimeoutError("ring deadlock: a rank never finished its step")
        for e in errors:
            if e:
                raise e
        return results

    def close(self) -> None:
        for s in self._socks:
            try:
                s.close()
            except OSError:
                pass


_WORLD: Optional[World] = None


def init(size: int = 4) -> World:
    """Start `size` local ranks wired as a ring. shutdown() reverses it."""
    global _WORLD
    assert 2 <= size <= 16, "lab-scale ring"
    w = World(size)
    # pre-assign outbound connections: connector side needs listener ports first
    listeners = {}
    for r in range(size):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        listeners[r] = s
    outs = {}
    for r in range(size):
        dst = (r + 1) % size
        c = socket.create_connection(listeners[dst].getsockname())
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        c.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 24)
        outs[r] = c
    ins = {}
    for r in range(size):
        conn, _ = listeners[r].accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 24)
        ins[r] = conn
    for r in range(size):
        w.ranks.append(RankCtx(r, size, outs[r], ins[r]))
    w._socks = list(outs.values()) + list(ins.values()) + list(listeners.values())
    for s in listeners.values():
        s.close()
    _WORLD = w
    return w


def shutdown() -> None:
    global _WORLD
    if _WORLD:
        _WORLD.close()
    _WORLD = None


def world() -> World:
    assert _WORLD is not None, "call ring.init() first"
    return _WORLD
