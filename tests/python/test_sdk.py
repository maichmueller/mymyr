"""The C++ SDK that the wheel carries: headers and the CMake package (mymyr.get_include / mymyr.get_cmake_dir)."""

import pathlib

import pytest

import mymyr

PACKAGE = pathlib.Path(mymyr.__file__).resolve().parent


@pytest.fixture(scope="module")
def sdk_installed():
    # an editable or in-tree install has no installed headers next to the package
    if not (PACKAGE / "include").is_dir():
        pytest.skip("mymyr is not an installed wheel: no SDK next to the package")


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
