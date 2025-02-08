# MRay

![sponza_teaser](sponza_teaser_1024spp.png)
*[Intel Sponza][6] with Curtains 1080p (Nvidia RTX 3070ti Mobile, 1024spp / 40.41ms spp )*

MRay: An interactive GPU-Based Renderer / Framework. MRay is extensible and performant ray tracing-based renderer framework written in CUDA.

> [!Note]
> This readme is incomplete I will improve it in future. Please check the code and .json files for configuration and scene definition.

## Building

### Dependencies

Most of the dependencies are handled by the build system itself via CMake external projects. User is responsible for some system libraries:

#### Common

- CMake. A recent version of CMake should suffice. Tested with v3.30.0.
- C++20 compliant C++ compiler. Only tested with latest MSVC and Clang-18.
- CUDA Installation. Version 12 and above is recommended.
- Nvidia OptiX installation (Required for HW accelerated ray tracing, which is strongly recommended)
- For documentation a python installation with these packages are required:
    - Sphinx
    - myst_parser
    - sphinx_book_theme
    - sphinx_copybutton

#### Windows
 Besides the common packages building the project in Windows do not require and extra packages. Vulkan-SDK is recommended for validation layer for debugging. We do not auto build/install the entire vulkan sdk but only the headers and libraries.

#### Linux (apt package names)
- libtbb-dev (required for parallel execution of c++ std library)
- xorg-dev (for Visor, dependency of glfw)
- libx11-dev (for Visor, dependency of glfw)

#### Automatically managed libraries

These libraries are automatically managed by the build system (uncommon libraries have their link):
 - fmt
 - nlohmann_json
 - [bs_thread_pool][1]
 - assimp
 - [gfg][2]
 - spdlog
 - CLI11
 - glfw [^1]
 - vulkan_headers [^1]
 - vulkan_loader [^1]
 - imgui [^1]
 - SLang [^1]
 - googletest [^2]
 - OIIO
    - zlib
    - OpenEXR-imath
    - OpenEXR
    - libtiff
    - libjpeg-turbo
    - libpng

[^1]: Only when Visor is built.

[^2]: Only when Tests are built.

### CMake Build

Build system is somewhat principled. External projects will be installed into `Ext` folder in the project's root directory. `Lib` directory will contain built packages in a platform/configuration specific manner.

We recommend CMake output directory `Bin/CMake` but it should work with any out-of-source build location **inside the project's root directory**. Final binaries and shaders (if applicable) will be written `Bin/{platform_name}/{configuration}`. This is your installation as well. You can directly copy the contents of this folder if you like (You can delete the `.lib` / `.a` files).

Only tested `ninja` and `Visual Studio Solution` platform configurations.

#### CMake Flags

These flags are not required to be set and all of which has a default value. `MRAY_ENABLE_HW_ACCELERATION` is highly recommended yo utilize GPU's native ray tracing capabilities.

- `CMAKE_CUDA_ACHITECTURES`: Select the CUDA SM level for compilation (default: "native").
- `MRAY_BUILD_DOCS`: Generate documentation build command (default: false).
- `MRAY_BUILD_TESTS`: Generate test projects (default: false).
- `MRAY_BUILD_VISOR`: Enable Visor project. Visor is real-time image viewer/tone mapper for the renderer (for interactive rendering)

- `MRAY_ENABLE_HW_ACCELERATION`: Enables GPU Hardware acceleration for ray tracing. CMake script look common locations for the required libraries
(only CUDA/OptiX at the moment).
    - `OPTIX_INSTALL_DIR`: If script could not find the required library location, you need to manually set this variable (i.e. `C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0`).

- `MRAY_ENABLE_PCH`: Enable precompiled headers. It does not benefit the build much since the main bottleneck is nvcc which do not support precompiled headers (default: false).

- `MRAY_EXPORT_COMPILE_COMMANDS`: Exports compile commands for lsp and include-what-you-use purposes.

- `MRAY_DEVICE_BACKEND`: Select the device backend. Currently only CUDA is supported. Hopefully we will add more backends in future. (default: "MRAY_GPU_BACKEND_CUDA").

- `MRAY_HOST_ARCH`: Hint the compiler for the underlying host architecture. Supported values are "MRAY_HOST_ARCH_BASIC", "MRAY_HOST_ARCH_AVX2" "MRAY_HOST_ARCH_AVX512" (default: "MRAY_HOST_ARCH_BASIC").

After configuring the project, you need to first run `MRayExternal` project/target. This target will generate automatically managed libraries/executables. After that building the solution / or "all" target will generate the binaries.

> [!Note]
> `MRayExternal` target specifically exempt from the "all" target. It takes considerable amount of time (especially in Windows) just to check if the target is up to date. You need to explicitly build this target first.

## Usage

Please install some example scenes from [here][3].

After build is complete, run either of these commands from the `WorkingDir` directory.

> `/path/to/MRay.exe visor -r 1920x1080 -s Scenes/CrySponza/crySponza.json  --tConf tracerConfig.json --rConf renderConfig.json --vConf visorConfig.json`

Which will run the interactive renderer. Some simple shortcuts:
 - `M` toggles the top bar.
 - `N` toggles the bottom bar.
 - `Numpad 6` and `Numpad 4` changes the scene's camera.
 - `Numpad 9` and `Numpad 7` changes the renderer.
 - `Numpad 3` and `Numpad 1` renderer specific logic 0.
 - `Numpad +` and `Numpad -` renderer specific logic 1.

For example; `Numpad 3` and `Numpad 1` in "SurfaceRenderer" toggle between AO/Furnace Test/Normal/Position sub-renderers.

 - `[` and `]` changes the movement scheme (FPS-like or Modelling software-like input). `WASD` to move left mouse button + mouse movement to travel through the scene. Holding `left shift` will increase the speed.
 - `Numpad 5` locks/unlocks the movement.
 - `P` pauses the rendering, `O` starts/stops the rendering.
 - `Escape` closes the window
 - `G` saves the image as an SDR Image.
 - `H` saves the image as an HDR Image.

If you generate image without the GUI run:

> `/path/to/MRay.exe run -r 1920x1080 -s Scenes/CrySponza/crySponza.json  --tConf tracerConfig.json --rConf renderConfig.json`

## Documentation

> [!Warning]
> Documentation is not yet available! (TODO!)

## Future Work
 - Implement a "host" renderer (CPU).
 - Implement "device" abstraction layers for Intel and AMD gpus (via HIP and SYCL respectively). Functionality is abstracted away, so this should be relatively straightforward.
 - Add support for volume rendering
 - Add support for spectral rendering (Again abstraction is there but not correct at the moment it requires quite a bit of work).

## Similar Projects

#### Mitsuba3

[https://github.com/mitsuba-renderer/mitsuba3][4]

#### pbrt-v4

[https://github.com/mmp/pbrt-v4][5]

## License

MRay is under the Apache 2.0 License. See [LICENSE][lic] for details.

[1]: https://github.com/bshoshany/thread-pool
[2]: https://github.com/yalcinerbora/GFGFileFormat
[3]: https://drive.google.com/file/d/1yPcrjZuyCDQVhFWHN0ZqMdeOpWn7WR76/view?usp=sharing
[4]: https://github.com/mitsuba-renderer/mitsuba3
[5]: https://github.com/mmp/pbrt-v4
[6]: https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html
[lic]: LICENSE