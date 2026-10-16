"""The C++ SDK that the wheel and an editable install carry: headers and the CMake package (mymyr.get_include /
mymyr.get_cmake_dir), next to the compiled extension."""

import pathlib

import pytest

import mymyr

INSTALL = pathlib.Path(mymyr._core.__file__).resolve().parent


@pytest.fixture(scope="module")
def sdk_installed():
    # an in-tree build has no installed headers next to the extension
    if not (INSTALL / "include").is_dir():
        pytest.skip("mymyr is an in-tree build: no SDK next to the extension")


def test_get_include(sdk_installed):
    include = pathlib.Path(mymyr.get_include())
    assert include.is_dir()
    assert (include / "mymyr" / "ext.h").is_file()
    assert (include / "mymyr" / "search" / "iw.hpp").is_file()
    assert (include / "mymyr" / "version.hpp").is_file()


def test_get_cmake_dir(sdk_installed):
    cmake_dir = pathlib.Path(mymyr.get_cmake_dir())
    assert cmake_dir.is_dir()
    for name in ("mymyrConfig.cmake", "mymyrConfigVersion.cmake", "mymyrTargets.cmake"):
        assert (cmake_dir / name).is_file()


def test_the_libraries_are_installed(sdk_installed):
    lib = pathlib.Path(mymyr.get_cmake_dir()).parents[1]
    assert any(lib.glob("libmymyr_core.*"))
    if hasattr(mymyr, "Domain"):
        assert any(lib.glob("libmymyr_frontend.*"))


def test_the_paths_are_plain_strings():
    assert isinstance(mymyr.get_include(), str)
    assert isinstance(mymyr.get_cmake_dir(), str)


def test_the_paths_are_next_to_the_extension():
    assert pathlib.Path(mymyr.get_include()).parent == INSTALL
    assert pathlib.Path(mymyr.get_cmake_dir()).parents[2] == INSTALL
