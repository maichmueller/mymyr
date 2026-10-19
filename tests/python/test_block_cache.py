"""The cache of recycled output blocks: releasing its memory, views that outlive their batch, and fork."""

import gc
import os
import signal
import threading
import warnings

import numpy as np
import pytest

import mymyr
from mymyr import encoding
from conftest import text_task, walk

MIB = 2**20


def encoder_of(task):
    return encoding.FlatRelationEncoder(task, goal_derivations=())


def batch_bytes(batch):
    return sum(array.nbytes for array in batch.as_dict().values() if isinstance(array, np.ndarray))


class Workload:
    """A gripper task, the encoder and rows of the requested batch sizes in bytes."""

    def __init__(self):
        self.task = text_task("gripper__prob05")
        self.encoder = encoder_of(self.task)
        self.base = np.ascontiguousarray(self.task.encode(walk(self.task, steps=40, walks=4)))
        one = self.encoder.encode(self.base[:256])
        self.bytes_per_state = batch_bytes(one) / 256

    def rows(self, nbytes, shift=0):
        count = int(nbytes / self.bytes_per_state) + 1
        rows = np.resize(np.roll(self.base, shift, axis=0), (count, self.base.shape[1]))
        return np.ascontiguousarray(rows)


@pytest.fixture(scope="module")
def workload():
    return Workload()


def resident_bytes():
    try:
        with open("/proc/self/statm") as statm:
            return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        pytest.skip("no /proc/self/statm on this platform")


def test_the_cache_is_exposed_and_documented():
    assert isinstance(mymyr.block_cache_bytes(), int)
    assert mymyr.trim_block_cache() is None
    assert mymyr.block_cache_bytes() == 0
    for function in (mymyr.trim_block_cache, mymyr.block_cache_bytes):
        assert "block" in function.__doc__


def test_trim_returns_the_memory_of_released_batches(workload):
    rows = workload.rows(48 * MIB)
    mymyr.trim_block_cache()
    batch = workload.encoder.encode(rows)
    size = batch_bytes(batch)
    assert size > 32 * MIB
    del batch
    gc.collect()
    assert mymyr.block_cache_bytes() >= size  # the cache keeps the block, and with it the pages
    cached = resident_bytes()
    mymyr.trim_block_cache()
    assert mymyr.block_cache_bytes() == 0
    assert cached - resident_bytes() >= size // 2


def test_a_one_off_large_batch_does_not_stay_resident(workload):
    one_off = workload.rows(48 * MIB)
    small = workload.rows(MIB)
    mymyr.trim_block_cache()
    batch = workload.encoder.encode(one_off)
    size = batch_bytes(batch)
    del batch
    gc.collect()
    assert mymyr.block_cache_bytes() >= size
    resident = resident_bytes()
    for _ in range(300):  # each acquires a block of more than 128 KiB; none of them takes the large one
        workload.encoder.encode(small)
    assert mymyr.block_cache_bytes() < 8 * MIB
    assert resident - resident_bytes() >= size // 2


@pytest.mark.parametrize("framework", ["numpy", "dlpack"])
@pytest.mark.parametrize("nbytes", [MIB // 4, 3 * MIB, 24 * MIB])
def test_views_outlive_their_batch_and_encoder_while_blocks_recycle(workload, framework, nbytes):
    task = workload.task
    rows = workload.rows(nbytes)
    others = [workload.rows(nbytes, shift=shift) for shift in (1, 2, 3)]
    encoder = encoder_of(task)
    batch = encoder.encode(rows)
    expected = {name: array.copy() for name, array in batch.as_dict().items() if isinstance(array, np.ndarray)}
    exported = batch.as_dict(framework)
    views = {name: np.from_dlpack(exported[name]) if framework == "dlpack" else exported[name] for name in expected}
    del batch, exported, encoder
    gc.collect()
    other_encoder = encoder_of(task)
    for _ in range(3):  # blocks of the same size class are acquired, written and released meanwhile
        for rows_of_other in others:
            other_encoder.encode(rows_of_other)
    for name, array in expected.items():
        assert np.array_equal(views[name], array), name
    del views
    gc.collect()
    again = other_encoder.encode(rows)  # the views' block is recycled now and holds the batch again
    for name, array in expected.items():
        assert np.array_equal(again.as_dict()[name], array), name


def test_a_forked_child_encodes_large_batches_while_another_thread_uses_the_cache(workload):
    if not hasattr(os, "fork"):
        pytest.skip("no fork on this platform")
    rows = workload.rows(12 * MIB)
    expected = workload.encoder.encode(rows).relation_args.copy()
    stop = threading.Event()
    failures = []

    def churn():
        try:
            while not stop.is_set():
                workload.encoder.encode(rows)
        except BaseException as error:  # noqa: BLE001 - reported by the test
            failures.append(error)

    thread = threading.Thread(target=churn)
    thread.start()
    try:
        for _ in range(8):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # JAX warns about every fork once it is loaded
                # CPython warns about a fork of a multi-threaded process; this fork is the case under test
                warnings.filterwarnings("ignore", message=".*multi-threaded.*fork", category=DeprecationWarning)
                child_id = os.fork()
            if child_id == 0:
                status = 1
                try:
                    signal.alarm(30)  # a child that waits on a lock it inherited ends here
                    child = workload.encoder.encode(rows)
                    status = 0 if np.array_equal(child.relation_args, expected) else 2
                finally:
                    os._exit(status)
            _, wait_status = os.waitpid(child_id, 0)
            assert os.waitstatus_to_exitcode(wait_status) == 0
    finally:
        stop.set()
        thread.join()
    assert not failures
