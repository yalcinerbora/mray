# Building

Building the project is as streamlined as possible. However; it is still cumbersome to my taste. System dependencies are minimized, and only cpp compiler, CMake, git and device compiler (CUDA) should be provided by the system. Rest will be downloaded and built automatically. Unfortunately, this mandates working internet connection while building the binaries for the first time.

You do not need `--recursive` flag while cloning the repository. For automatic dependency downloading and building, system relies on CMake's 'ExternalProject_Add' functionality.

## User-Responsible Dependencies

As stated above, most of the dependencies are handled by the build system itself via CMake external projects. User is responsible for some system libraries:

### Common

- CMake (mimimum v3.26). A recent version of CMake should suffice. Tested with v3.30.0.
- Git. Version should not matter (hopefully)
- C++20 compliant (except for modules) C++ compiler. Only tested with latest MSVC, Clang-18/19 and gcc-13 (gcc-13 has quite a bit of warnings which are not fixed yet).
- CUDA Installation. Version 12 and above is recommended.
- Nvidia OptiX (minimum v8.0) installation (Required for HW accelerated ray tracing, which is strongly recommended)
- For documentation, a python installation with these packages are required:
    - Sphinx
    - myst_parser
    - sphinx_book_theme
    - sphinx_copybutton

:::{important}
There is a bug on recent CUDA v12.9 which prevents compilation of the std::optional with 16-byte aligned types. This is only the case for MSVC-STL. Up until then v12.6 is recommended for MSVC builds.
:::

### Windows
 Besides the common packages, building the project in Windows should not require and extra packages. Vulkan-SDK is recommended for validation layer for debugging. We do not auto build/install the entire vulkan sdk but only the headers and the loader.

### Linux (apt package names)
- xorg-dev (for Visor, dependency of glfw)
- libx11-dev (for Visor, dependency of glfw)

## Automatically managed libraries

These libraries are automatically managed by the build system (uncommon libraries have their link):
 - fmt
 - nlohmann_json
 - assimp
 - [gfg](https://github.com/yalcinerbora/GFGFileFormat)
 - spdlog
 - CLI11
 - glfw [1]
 - vulkan_headers [1]
 - vulkan_loader [1]
 - imgui [1]
 - SLang [1]
 - googletest [2]
 - OIIO
    - zlib
    - OpenEXR-imath
    - OpenEXR
    - libtiff
    - libjpeg-turbo
    - libpng
    - OpenColorIO
        - expat
        - minizip_ng
        - yaml-cpp
        - pystring
- OpenUSD [3]
    - oneTBB
- tracy [4]
- Embree [5]

:::{container} footer-block

\[1\]: Only when Visor is built.

\[2\]: Only when Tests are built.

\[3\]: Only when OpenUSD support is enabled.

\[4\]: Only when tracy support is enabled.

\[5\]: Only when CPU-backend is built and `MRAY_ENABLE_HW_ACCELERATION` CMake variable is set.

---
:::

:::{Note}

### Why automatic management? I have most of these libraries, can't I use these?

Unfortunately you. W it is **37 GB** of stuff.

:::

## CMake Build

Build system is somewhat principled. External projects will be installed into `Ext` folder in the project's root directory. `Lib` directory will contain built packages in a platform/configuration specific manner.

We recommend CMake output directory `Bin/CMake` but it should work with any out-of-source build location **inside the project's root directory**. Final binaries and shaders (if applicable) will be written `Bin/{platform_name}/{configuration}`. This is your installation as well. You can directly copy the contents of this folder if you like (You can delete the `.lib` / `.a` files).

Only tested `ninja` and `Visual Studio Solution` platform configurations.

:::{note}
While configuring the project, you need to set `CMAKE_C_COMPILER` as well as a `CMAKE_CXX_COMPILER`. Although project does not use any pure C code, this flag will be propagated to the external projects that are written in C (such as libtiff). You may need to set `CMAKE_CUDA_COMPILER` on Linux platforms if nvcc is not in your `PATH`.
:::

### CMake Flags

:::{table} CMake Parameters of the Build System
:widths: auto
:align: center

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `CMAKE_CUDA_ACHITECTURES`      | `string`{sup}`1` | "native"  | Select the CUDA SM level for compilation. |
| `MRAY_BUILD_DOCS`              | `boolean`        | off       | Generate documentation build project. |
| `MRAY_BUILD_TESTS`             | `boolean`        | off       | Generate test projects (default: false). |
| `MRAY_BUILD_VISOR`             | `boolean`        | on        | Enable Visor project. Visor is real-time image viewer/tone mapper for the renderer (for interactive rendering.) |
| `MRAY_ENABLE_USD`              | `boolean`        | off       | Enable OpenUSD support. MRay will try to load the USD scene. (**Experimental**) |
| `MRAY_ENABLE_HW_ACCELERATION`  | `boolean`        | off       | Enables GPU Hardware acceleration for ray tracing. CMake script look common locations for the required. |
| `MRAY_ENABLE_PCH`              | `boolean`        | off       | Enable precompiled headers. It does not benefit the build much since the main bottleneck is nvcc which do not support precompiled headers. |
| `MRAY_EXPORT_COMPILE_COMMANDS` | `boolean`        | on        | Exports compile commands for lsp and include-what-you-use purposes. |
| `MRAY_DEVICE_BACKEND`          | `string`{sup}`2` | "MRAY_GPU_BACKEND_CUDA" | Select the device backend. Currently only CUDA is supported. Hopefully we will add more backends in future. |
| `MRAY_HOST_ARCH`               | `string`{sup}`3` | "MRAY_HOST_ARCH_BASIC"  | Hint the compiler for the underlying host architecture. |
| `MRAY_BUILD_HOST_BACKEND`      | `boolean`        | off       | Creates a CPU target that emulates GPU via custom thread pool. Since it runs GPU-optimized code, its performance may not competitive (compared to other CPU-based renderers). |
| `MRAY_DISABLE_DEVICE_BACKEND`  | `boolean`        | off       | Do not create build targets for selected device backend. This flag makes sense only in conjunction with `MRAY_BUILD_HOST_BACKEND`. |
| `MRAY_SANITIZER_MODE`          | `string`{sup}`4` | "address" | Mode of the SanitizeR build configuration. Supported values differ depending on the cpp compiler. For MSVC: only "address" is supported. For gcc and clang: all supported -fsanitize modes should be available (default: "address").
| `MRAY_ENABLE_TRACY`            | `boolean`        | off       | Enable tracy profiling support. MRay.exe will accept -p flag for profiling. |
| `MRAY_COMPILE_OPTIX_AS_PTX`    | `boolean`        | off       | Compile OptiX shim layer shaders as PTX. This may be useful sometimes when shaders fails to compile with optixir.  |

:::

:::{container} footer-block

\[1\]: `native`, `52`, `60`, `61`, `70`, `72`, `75`, `86`, `89`, `90`, `all` or `all-major`

\[2\]: `MRAY_GPU_BACKEND_CUDA`

\[3\]: `MRAY_HOST_ARCH_BASIC`, `MRAY_HOST_ARCH_AVX2` or `MRAY_HOST_ARCH_AVX512`

\[4\]: `address`, `undefined`, `memory` or `thread`

---
:::


We recommend enabling `MRAY_ENABLE_HW_ACCELERATION` to utilize GPU's native ray tracing capabilities. When it is enabled and `MRAY_DEVICE_BACKEND` is "MRAY_GPU_BACKEND_CUDA", you may need to set `OPTIX_INSTALL_DIR` CMake configuration parameter as well (i.e. `C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0`). 'FindOptiX.cmake' script will try to find OptiX automatically via looking at default installation locations.

After configuring the project, you need to first run `MRayExternal` project/target. This target will generate automatically managed libraries/executables. After that, building the solution (Visual Studio) or running the "all" target (Ninja) will generate the binaries.

:::{important}

`MRayExternal` target specifically exempt from the "all" target. It takes considerable amount of time (especially in Windows) just to check if the target is up to date. You need to explicitly build this target first and re-run it when you change the configuration that requires additional dependencies.

Additionally, you need to run `MRayExternal` for both "Release" and "Debug" configurations. There is a 3rd configuration "SanitizeR" which compiles the project with a sanitizer support by checking `MRAY_SANITIZER_MODE` CMake parameter. SanitizeR compiled with most of the optimizations and uses Release target's external dependencies.

:::