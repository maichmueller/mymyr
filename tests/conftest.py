"""Marks everything under tests/parity/python with the `parity` marker, so that `pytest -m "not parity"` runs only the
tests that need no reference data."""

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "tests/parity/python/" in item.path.as_posix():
            item.add_marker(pytest.mark.parity)
