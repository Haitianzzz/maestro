"""Unit tests for ``maestro.utils.priority_queue`` (spec 04 §7)."""

from __future__ import annotations

import asyncio

import pytest

from maestro.utils.priority_queue import AsyncPriorityQueue, PrioritySemaphore


async def _yield(times: int = 5) -> None:
    """Advance the event loop by ``times`` ticks."""
    for _ in range(times):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# AsyncPriorityQueue
# ---------------------------------------------------------------------------


async def test_priority_queue_single_waiter_runs_immediately() -> None:
    q = AsyncPriorityQueue()
    await q.acquire_slot(priority=0)
    # No other waiter; release should not explode.
    q.release_slot()
    assert q.pending == 0


async def test_priority_queue_higher_priority_runs_before_lower() -> None:
    q = AsyncPriorityQueue()
    order: list[str] = []

    async def worker(name: str, priority: int) -> None:
        await q.acquire_slot(priority=priority)
        order.append(name)
        q.release_slot()

    # Seed one "running" task so subsequent waiters must queue.
    await q.acquire_slot(priority=100)

    low = asyncio.create_task(worker("low", priority=0))
    high = asyncio.create_task(worker("high", priority=10))
    medium = asyncio.create_task(worker("medium", priority=5))
    await _yield()

    # Release the seed — waiters drain in priority order.
    q.release_slot()
    await asyncio.gather(low, high, medium)
    assert order == ["high", "medium", "low"]


async def test_priority_queue_fifo_within_same_priority() -> None:
    q = AsyncPriorityQueue()
    order: list[str] = []

    async def worker(name: str) -> None:
        await q.acquire_slot(priority=1)
        order.append(name)
        q.release_slot()

    await q.acquire_slot(priority=100)  # seed

    tasks = [asyncio.create_task(worker(name)) for name in ("a", "b", "c", "d")]
    await _yield()
    q.release_slot()
    await asyncio.gather(*tasks)
    assert order == ["a", "b", "c", "d"]


async def test_priority_queue_release_without_acquire_raises() -> None:
    q = AsyncPriorityQueue()
    with pytest.raises(RuntimeError, match="without a matching acquire_slot"):
        q.release_slot()


async def test_priority_queue_cancelled_waiter_does_not_block_others() -> None:
    """A waiter cancelled mid-wait must not steal the next slot (spec 04 §4)."""
    q = AsyncPriorityQueue()
    order: list[str] = []
    enqueued: dict[str, asyncio.Event] = {name: asyncio.Event() for name in "ABC"}
    finished = asyncio.Event()

    async def worker(name: str, priority: int) -> None:
        enqueued[name].set()
        await q.acquire_slot(priority=priority)
        try:
            order.append(name)
        finally:
            q.release_slot()

    # Seed one holder so subsequent arrivals queue.
    await q.acquire_slot(priority=100)

    a = asyncio.create_task(worker("A", priority=1))
    b = asyncio.create_task(worker("B", priority=5))
    c = asyncio.create_task(worker("C", priority=3))

    # Wait until every worker has pushed itself onto the heap.
    await asyncio.gather(*(e.wait() for e in enqueued.values()))
    # Give the queue a tick so the pushes are visible.
    await asyncio.sleep(0)
    assert q.pending == 3

    # Cancel B while it is waiting (event not set).
    b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await b

    # Release the seed; B is a dead entry, so the wake chain should serve C
    # (next-highest priority) then A — never B.
    q.release_slot()
    finished.set()
    await asyncio.gather(a, c)
    assert order == ["C", "A"]


# ---------------------------------------------------------------------------
# PrioritySemaphore basics
# ---------------------------------------------------------------------------


async def test_semaphore_rejects_zero_max_concurrent() -> None:
    with pytest.raises(ValueError, match="max_concurrent"):
        PrioritySemaphore(0)


async def test_semaphore_immediate_acquire_when_free() -> None:
    sem = PrioritySemaphore(2)
    async with sem.acquire(priority=0):
        assert sem.in_flight == 1
        assert sem.waiting == 0
    assert sem.in_flight == 0


async def test_semaphore_limits_concurrency() -> None:
    sem = PrioritySemaphore(2)
    in_flight_peak = 0
    current = 0
    gate = asyncio.Event()

    async def worker() -> None:
        nonlocal current, in_flight_peak
        async with sem.acquire(priority=0):
            current += 1
            in_flight_peak = max(in_flight_peak, current)
            await gate.wait()
            current -= 1

    tasks = [asyncio.create_task(worker()) for _ in range(5)]
    await _yield()
    assert in_flight_peak <= 2
    assert sem.in_flight == 2
    assert sem.waiting == 3
    gate.set()
    await asyncio.gather(*tasks)
    assert in_flight_peak == 2
    assert sem.in_flight == 0


# ---------------------------------------------------------------------------
# Priority semantics under contention
# ---------------------------------------------------------------------------


async def test_semaphore_higher_priority_overtakes_lower() -> None:
    sem = PrioritySemaphore(1)
    order: list[str] = []
    gate = asyncio.Event()

    async def occupier() -> None:
        async with sem.acquire(priority=100):
            order.append("occupier")
            await gate.wait()

    async def waiter(name: str, priority: int) -> None:
        async with sem.acquire(priority=priority):
            order.append(name)

    occ = asyncio.create_task(occupier())
    await _yield()  # let the occupier grab the slot
    assert sem.in_flight == 1

    low = asyncio.create_task(waiter("low", 0))
    high = asyncio.create_task(waiter("high", 10))
    mid = asyncio.create_task(waiter("mid", 5))
    await _yield()  # let them queue

    gate.set()
    await asyncio.gather(occ, low, high, mid)
    assert order == ["occupier", "high", "mid", "low"]


async def test_semaphore_fifo_within_same_priority() -> None:
    sem = PrioritySemaphore(1)
    order: list[str] = []
    gate = asyncio.Event()

    async def occupier() -> None:
        async with sem.acquire(priority=100):
            await gate.wait()

    async def waiter(name: str) -> None:
        async with sem.acquire(priority=5):
            order.append(name)

    occ = asyncio.create_task(occupier())
    await _yield()

    tasks = [asyncio.create_task(waiter(n)) for n in ("a", "b", "c", "d")]
    await _yield()

    gate.set()
    await asyncio.gather(occ, *tasks)
    assert order == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# Release / error paths
# ---------------------------------------------------------------------------


async def test_semaphore_releases_on_exception() -> None:
    sem = PrioritySemaphore(1)

    with pytest.raises(RuntimeError, match="boom"):
        async with sem.acquire(priority=0):
            raise RuntimeError("boom")

    assert sem.in_flight == 0
    # Second acquire must succeed afterwards.
    async with sem.acquire(priority=0):
        assert sem.in_flight == 1


async def test_semaphore_concurrent_release_drains_in_priority_order() -> None:
    sem = PrioritySemaphore(2)
    order: list[str] = []
    gate = asyncio.Event()

    async def occupier() -> None:
        async with sem.acquire(priority=100):
            await gate.wait()

    async def waiter(name: str, priority: int) -> None:
        async with sem.acquire(priority=priority):
            order.append(name)

    occ1 = asyncio.create_task(occupier())
    occ2 = asyncio.create_task(occupier())
    await _yield()
    assert sem.in_flight == 2

    lo = asyncio.create_task(waiter("lo", 0))
    hi = asyncio.create_task(waiter("hi", 10))
    md = asyncio.create_task(waiter("md", 5))
    await _yield()

    gate.set()
    await asyncio.gather(occ1, occ2, lo, hi, md)
    # Only lo/hi/md entries are in order; the two occupiers didn't append.
    # hi must precede md, and md must precede lo.
    assert order.index("hi") < order.index("md") < order.index("lo")


# ---------------------------------------------------------------------------
# Cancellation safety
# ---------------------------------------------------------------------------


async def test_semaphore_cancelled_waiter_in_heap_does_not_block_others() -> None:
    """A waiter cancelled mid-wait must not steal a future slot.

    Scenario: one task occupies the single slot; three waiters queue at
    different priorities; the highest-priority waiter (B) is cancelled while
    it is blocked in ``event.wait()``. When the occupier releases, the slot
    must go to C (next-highest priority), not be silently consumed by B.
    """
    sem = PrioritySemaphore(1)
    order: list[str] = []
    occ_holds = asyncio.Event()
    occ_release = asyncio.Event()
    enqueued: dict[str, asyncio.Event] = {name: asyncio.Event() for name in "ABC"}

    async def occupier() -> None:
        async with sem.acquire(priority=100):
            occ_holds.set()
            await occ_release.wait()

    async def waiter(name: str, priority: int) -> None:
        enqueued[name].set()
        async with sem.acquire(priority=priority):
            order.append(name)

    occ = asyncio.create_task(occupier())
    await occ_holds.wait()
    assert sem.in_flight == 1

    a = asyncio.create_task(waiter("A", priority=1))
    b = asyncio.create_task(waiter("B", priority=5))
    c = asyncio.create_task(waiter("C", priority=3))

    # Wait until all three have reached the acquire call.
    await asyncio.gather(*(e.wait() for e in enqueued.values()))
    await asyncio.sleep(0)  # let the pushes settle
    assert sem.waiting == 3

    # Cancel B while it is still blocked in event.wait() (event not set).
    b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await b

    # Release the occupier; the wake chain must skip dead B and serve C
    # (next-highest priority), then A on the subsequent release.
    occ_release.set()
    await asyncio.gather(occ, a, c)
    assert order == ["C", "A"]
    assert sem.in_flight == 0
    assert sem.waiting == 0


async def test_semaphore_cancelled_after_grant_releases_slot() -> None:
    """Grant-cancel race must not leak the handed-out slot.

    Scenario: one task holds the slot; a single waiter W is queued; we set
    the gate (so the occupier eventually calls ``_release`` → ``_try_wake``
    sets W's event AND increments ``in_flight``) and, in the same event-loop
    tick, ``cancel`` on W. W's ``event.wait()`` may raise CancelledError
    with ``event.is_set() == True``; our handler must detect this and
    release the slot explicitly so the semaphore doesn't deadlock.
    """
    sem = PrioritySemaphore(1)
    occ_holds = asyncio.Event()
    occ_release = asyncio.Event()
    doomed_enqueued = asyncio.Event()

    async def occupier() -> None:
        async with sem.acquire(priority=100):
            occ_holds.set()
            await occ_release.wait()

    async def doomed() -> None:
        doomed_enqueued.set()
        async with sem.acquire(priority=5):
            # If we ever reach the body, sleep so the test can observe.
            await asyncio.sleep(3600)

    occ = asyncio.create_task(occupier())
    await occ_holds.wait()

    doomed_task = asyncio.create_task(doomed())
    await doomed_enqueued.wait()
    await asyncio.sleep(0)
    assert sem.waiting == 1

    # Schedule release and cancel back-to-back. Depending on event-loop
    # scheduling, the cancel may land either before the grant (waiter just
    # cancelled, heap skip) or after (grant-cancel race, explicit release
    # path). Either way the invariant below must hold.
    occ_release.set()
    doomed_task.cancel()

    results = await asyncio.gather(occ, doomed_task, return_exceptions=True)
    assert any(isinstance(r, asyncio.CancelledError) for r in results)
    assert sem.in_flight == 0
    assert sem.waiting == 0

    # Semaphore must still be usable: a fresh acquire succeeds immediately.
    async with sem.acquire(priority=0):
        assert sem.in_flight == 1
    assert sem.in_flight == 0
