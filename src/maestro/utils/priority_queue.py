"""Priority-aware concurrency primitives (spec 04 §4).

``asyncio.Semaphore`` limits concurrency but grants slots in FIFO order.
Maestro's scheduler needs *priority* scheduling so a critical-path subtask
(one that blocks downstream batches) runs ahead of a lower-priority cosmetic
task when slots are scarce.

This module provides two primitives:

* :class:`AsyncPriorityQueue` — a bare wait-queue keyed on
  ``(priority, fifo_seq)`` with no concurrency cap. Kept public because
  spec 04 §4.2 names it.
* :class:`PrioritySemaphore` — a priority-respecting async semaphore with a
  hard concurrency limit. Use via ``async with sem.acquire(priority=...):``.

Semantics
---------

* ``priority`` is an ``int``. Higher values are served first.
* Within the same priority, waiters are served in FIFO arrival order.
* Every call to :meth:`PrioritySemaphore.acquire` enqueues first, then
  attempts to wake the heap head. That way if a free slot and a concurrent
  higher-priority acquire arrive in the same event-loop tick, the
  higher-priority caller still overtakes.

Cancellation safety
-------------------

If the caller's task is cancelled while it is waiting for its priority
slot, we must not leave a live ``_Waiter`` behind on the heap — otherwise
a future ``_wake_head`` / ``_try_wake`` would hand a slot to a dead
awaiter and leak concurrency capacity (eventually dead-locking the
semaphore).

Two cases are handled explicitly:

* Cancel arrives while ``event.wait()`` is still blocked (event not set):
  the waiter is marked ``cancelled=True`` and stays on the heap. The wake
  path skips cancelled entries without consuming a slot.
* Cancel arrives in the same tick as the grant (``event.wait()`` raises
  ``CancelledError`` but ``event.is_set()`` is already ``True``): a slot
  was handed out before the exception propagated, so we explicitly
  release it before re-raising.

Thread-safety
-------------

These primitives are **not** thread-safe. They rely on asyncio's
single-threaded cooperative scheduler: the critical sections between
``await`` points cannot be interrupted. This matches Maestro's
asyncio-only concurrency model (DESIGN §3.6).
"""

from __future__ import annotations

import asyncio
import heapq
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass(order=True)
class _Waiter:
    """Heap entry ordered by ``(-priority, fifo_seq)``.

    Python's ``heapq`` is a min-heap, so we negate priority to make higher
    priority numbers bubble to the top. ``fifo_seq`` breaks ties in arrival
    order. ``event`` and ``cancelled`` do not participate in ordering; they
    are mutated after enqueue to communicate state to the wake path.
    """

    neg_priority: int
    fifo_seq: int
    event: asyncio.Event = field(compare=False)
    cancelled: bool = field(default=False, compare=False)


class AsyncPriorityQueue:
    """Priority queue of async waiters (higher priority number = earlier).

    The queue has no concurrency cap of its own; :meth:`acquire_slot` waits
    until the caller is at the head AND :meth:`release_slot` has been called
    by someone else (or the caller is the first arrival). This is the raw
    primitive that :class:`PrioritySemaphore` builds on.
    """

    def __init__(self) -> None:
        self._heap: list[_Waiter] = []
        self._seq = 0
        # Number of slots currently "handed out" — incremented when we wake
        # a live waiter, decremented by release_slot.
        self._granted = 0

    def __len__(self) -> int:
        """Total waiters + granted slots, for diagnostics."""
        return len(self._heap) + self._granted

    @property
    def pending(self) -> int:
        return len(self._heap)

    async def acquire_slot(self, priority: int) -> None:
        """Wait until no other waiter is ahead of us by priority.

        An "unlimited" slot is granted once we are admitted. Callers must
        call :meth:`release_slot` exactly once afterwards. Cancellation is
        safe — see module docstring "Cancellation safety".
        """
        seq = self._seq
        self._seq += 1
        event = asyncio.Event()
        waiter = _Waiter(neg_priority=-priority, fifo_seq=seq, event=event)
        heapq.heappush(self._heap, waiter)

        # If nobody is currently running, wake the (priority-max) head.
        if self._granted == 0:
            self._wake_head()

        try:
            await event.wait()
        except BaseException:
            if event.is_set():
                # We were granted a slot before the cancel fired; hand it back.
                self.release_slot()
            else:
                waiter.cancelled = True
            raise

    def release_slot(self) -> None:
        """Give up the slot and wake the next-highest-priority waiter."""
        if self._granted <= 0:
            raise RuntimeError("release_slot called without a matching acquire_slot")
        self._granted -= 1
        self._wake_head()

    def _wake_head(self) -> None:
        """Pop cancelled waiters without consuming slots; wake the first live one."""
        while self._heap:
            head = heapq.heappop(self._heap)
            if head.cancelled:
                # Dead entry; skip without incrementing granted.
                continue
            self._granted += 1
            head.event.set()
            return


class PrioritySemaphore:
    """Bounded async semaphore that respects waiter priority.

    Higher ``priority`` integers win over lower ones. Within the same
    priority, FIFO arrival order is preserved.
    """

    def __init__(self, max_concurrent: int) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._max = max_concurrent
        self._in_flight = 0
        self._heap: list[_Waiter] = []
        self._seq = 0

    @property
    def max_concurrent(self) -> int:
        return self._max

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def waiting(self) -> int:
        return len(self._heap)

    @asynccontextmanager
    async def acquire(self, priority: int = 0) -> AsyncIterator[None]:
        """Acquire a slot respecting priority; release on context exit."""
        await self._acquire(priority)
        try:
            yield
        finally:
            self._release()

    async def _acquire(self, priority: int) -> None:
        # Always enqueue first so a concurrent higher-priority caller in the
        # same event-loop tick can overtake us.
        seq = self._seq
        self._seq += 1
        event = asyncio.Event()
        waiter = _Waiter(neg_priority=-priority, fifo_seq=seq, event=event)
        heapq.heappush(self._heap, waiter)
        self._try_wake()
        try:
            await event.wait()
        except BaseException:
            if event.is_set():
                # Grant-cancel race: _try_wake already incremented in_flight
                # for us before the cancel fired. Release explicitly so the
                # slot is not leaked.
                self._release()
            else:
                # Cancel fired while we were still queued. Mark the entry
                # dead so _try_wake skips it rather than handing a slot to
                # a defunct awaiter.
                waiter.cancelled = True
            raise

    def _release(self) -> None:
        if self._in_flight <= 0:
            raise RuntimeError("release called without a matching acquire")
        self._in_flight -= 1
        self._try_wake()

    def _try_wake(self) -> None:
        """Wake as many live heap heads as there is spare capacity.

        Cancelled entries are popped without consuming slots — this is what
        prevents a cancelled waiter from permanently stealing a slot.
        """
        while self._heap and self._in_flight < self._max:
            head = heapq.heappop(self._heap)
            if head.cancelled:
                continue
            self._in_flight += 1
            head.event.set()


__all__ = ["AsyncPriorityQueue", "PrioritySemaphore"]
