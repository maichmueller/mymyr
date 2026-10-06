import sys
import threading

import numpy as np

import mymyr


def test_version_and_build():
    assert mymyr.__version__ == "0.1.0"
    assert mymyr.__version__ in mymyr.build_info()


def test_free_threaded_interpreter_stays_gil_free():
    # importing the extension must not re-enable the GIL on a free-threaded interpreter
    if hasattr(sys, "_is_gil_enabled"):
        assert mymyr.free_threaded_build()
        assert not sys._is_gil_enabled()


def test_hash_rows_ignores_trailing_zero_words():
    a = np.array([[1, 2, 0, 0], [1, 2, 3, 0], [0, 0, 0, 0]], dtype=np.uint64)
    b = np.array([[1, 2]], dtype=np.uint64)
    h = mymyr.hash_rows(a)
    assert h.dtype == np.uint64 and h.shape == (3,)
    assert h[0] == mymyr.hash_rows(b)[0]
    assert h[0] != h[1]


def test_hash_rows_from_many_threads():
    rng = np.random.default_rng(0)
    states = rng.integers(0, 2**63, size=(20000, 3), dtype=np.uint64)
    expected = mymyr.hash_rows(states)
    results = [None] * 16

    def work(i):
        results[i] = mymyr.hash_rows(states)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for r in results:
        assert np.array_equal(r, expected)
