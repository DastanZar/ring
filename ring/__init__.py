"""ring — NCCL-shaped collective ops on plain sockets.

No GPU, no MPI, no CUDA: `init()` spawns local ranks connected by TCP, and
each collective runs the real ring algorithm — chunk the tensor, pipeline it
around a logical ring with `send_right / recv_left`, reduce as you go. That
is what "NCCL ring allreduce" means, minus NVLink; the bus-bandwidth formula
`2(N−1)/N · S` you derive for NCCL here predicts the same curve with sockets.

Everything is in-process threads so `pip install ring && pytest` works
anywhere CI runs; the algorithm fidelity is the point, not bandwidth records.
"""
from .bootstrap import init, shutdown, world
from .rings import allgather, allreduce, broadcast, busbw, reduce_scatter

__all__ = ["init", "shutdown", "world", "allreduce", "allgather", "broadcast", "reduce_scatter", "busbw"]
