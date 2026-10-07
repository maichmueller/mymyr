# C++ package

An installed mymyr package exports CMake targets in the `mymyr::` namespace. `mymyr::core` contains the planning
core; `mymyr::frontend` adds PDDL parsing; `mymyr::cuda` is present when the package was built with CUDA. Request only
the components the consumer uses:

```cmake
find_package(mymyr CONFIG REQUIRED COMPONENTS core frontend)
target_link_libraries(app PRIVATE mymyr::core mymyr::frontend)
```

For a source install, build the project and run `cmake --install build --prefix <prefix>`. Set `mymyr_DIR` to
`<prefix>/lib/cmake/mymyr` when configuring a consumer. An installed wheel contains the same headers and CMake
package; `mymyr.get_include()` returns its header directory and `mymyr.get_cmake_dir()` returns its CMake package
directory. These paths may not contain the installed SDK in an editable or in-tree install.

The C++ interface is source-level, not a stable binary ABI. Consumers need C++26 (GCC 16 or Clang 22 with libc++) and
the standard library used to build the package, and should rebuild when mymyr changes. The plain-C interface in
[`mymyr/ext.h`](../include/mymyr/ext.h) is the stable binary interface.

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
