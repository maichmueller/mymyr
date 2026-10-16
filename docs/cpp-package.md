# C++ package

An installed mymyr package exports CMake targets in the `mymyr::` namespace. `mymyr::core` contains the planning
core; `mymyr::encoding` adds native relational observations; `mymyr::frontend` adds PDDL parsing; `mymyr::cuda` is present
when the package was built with CUDA. Request only
the components the consumer uses:

```cmake
find_package(mymyr CONFIG REQUIRED COMPONENTS core frontend)
target_link_libraries(app PRIVATE mymyr::core mymyr::frontend)
```

For a source install, build the project and run `cmake --install build --prefix <prefix>`. Set `mymyr_DIR` to
`<prefix>/lib/cmake/mymyr` when configuring a consumer. An installed wheel contains the same headers and CMake
package; `mymyr.get_include()` returns its header directory and `mymyr.get_cmake_dir()` returns its CMake package
directory. An editable install (`pip install -e .`) has them too, next to its compiled extension in
site-packages; a build tree used through `PYTHONPATH` has none.

The package's C++ headers and static libraries are a source-level interface. Build consumers with C++26 (GCC 16 or
Clang 22 with libc++) and the standard library used to build the package, then rebuild them when mymyr changes. The
plain-C interface in [`mymyr/ext.h`](../include/mymyr/ext.h) provides the stable binary interface.

The `mymyr::cuda` component is built with `-DMYMYR_CUDA=ON`, which needs CUDA 13.4 or newer and CMake 4.4 or newer: its
device code is C++23 (nvcc with the project's GCC 16 as host compiler), while the headers a consumer includes stay
C++26. Its binaries run on NVIDIA drivers 580 or newer; see the [CUDA page](cuda.md).

The checked-in [C++ consumer project](../tests/cmake_consumer/) uses `find_package`, links `mymyr::core` and
`mymyr::frontend`, parses a PDDL task, then prints “IW ... plan length ...” from its result. It also has a core-only
mode that reads a normalized task file. The [consumer check script](../ci/check_cmake_consumer.sh) configures, builds
and runs both modes. In a wheel install, pass `$(.venv/bin/python -c 'import mymyr; print(mymyr.get_cmake_dir())')` as
the package directory.

```python
import mymyr

print(mymyr.get_include())
print(mymyr.get_cmake_dir())
```
