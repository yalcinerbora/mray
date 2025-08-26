# Usage

A sample 'working directory` is available which contains example configuration and scene files. You can install it from [here](https://drive.google.com/file/d/1XaVyLcHrGB35qv1rpVHtiP7sUmS9HYEZ/view?usp=sharing).

After build is complete, run either of these commands from the downloaded directory:

> /path/to/MRay.exe visor -r 1920x1080 -s Scenes/CrySponza/crySponza.json  --tConf tracerConfig.json --rConf renderConfig.json --vConf visorConfig.json

Which will run the interactive renderer over the classic Crytek's Sponza scene.

Some simple shortcuts:
 - `M` toggles the top bar.
 - `N` toggles the bottom bar.
 - `Numpad 6` and `Numpad 4` changes the scene's camera.
 - `Numpad 9` and `Numpad 7` changes the renderer.
 - `Numpad 3` and `Numpad 1` renderer specific logic 0.
 - `Numpad +` and `Numpad -` renderer specific logic 1.

For example; `Numpad 3` and `Numpad 1` in "SurfaceRenderer" toggle between AO/Furnace Test/Normal/Position... sub-renderers.

 - `[` and `]` changes the movement scheme (FPS-like or Modelling software-like input). `WASD` to move left mouse button + mouse movement to travel through the scene. Holding `left shift` will increase the speed.
 - `Numpad 5` locks/unlocks the movement keys so that you don't reset a long render by mistake.
 - `P` pauses the rendering, `O` starts/stops the rendering.
 - `Escape` closes the window and terminates the process.
 - `G` saves the current image as an SDR Image.
 - `H` saves the current image as an HDR Image.
 - `Numpad ,` print the current camera orientation to the stdout.

If you want to generate an image without the GUI run:

> $ /path/to/MRay.exe run -r 1920x1080 -s Scenes/CrySponza/crySponza.json  --tConf tracerConfig.json --rConf renderConfig.json

You can feed visor config to the `run` sub-command, it will just ignore the argument. There are no shortcuts for the `run` sub-command, it will terminate rendering when renderer deemed it is finished (via its own configuration parameters such as `totalSPP` field).

You can check parameters of all commands via:

> $ /path/to/MRay.exe --help-all

Or options of specific commands by:

> $ /path/to/MRay.exe subcommand_name -h

## Configuration Files

MRay system uses config files to reduce command line arguments since most of the arguments are required to be explicit. All configuration files are jsonc similar to [MRay scene files](scene/mrayScene.md).

### Tracer Config

Tracer config has two mandatory fields that are called `TracerDLL` and `Parameters`.

:::{table} Tracer DLL Struct
:widths: auto
:align: center

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| name      | `string` | explicit | Path to the .dll (.so) file in the system. OS specific lookup rules will apply. |
| construct | `string` | explicit | Mangled function name that constructs the tracer |
| destruct  | `string` | explicit | Mangled function name that destroys the tracer |

:::

:::{table} Params Struct
:widths: auto
:align: center

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| seed | `int`| explicit | Meta seed that will be used to construct multiple RNGs. How would it be used depens on other parameters (such as `samplerType`) and the renderer. |
| acceleratorType | `string`{sup}`1` | explicit | Type of the accelerator(s) that will be constructed and used.
| samplerType | `int`| explicit | Hint to the tracer's renderer how many discrete 'things' should be processed at once. For path tracer, the 'things' are paths. For texture view renderer, it is pixels.|
| clampTexRes | `int`| explicit | Automatically resize textures if size is greater than this value. Aspect ratio is preserved. |
| globalTexColorSpace | `string`{sup}`2`| explicit | Automatically convert textures to this color space while loading. Renderer  |
| genMipmaps | `boolean`| explicit | Automatically generate mipmaps for textures. Actual mipmaps of a texture has precedence. |
| mipGenFilter | `struct`{sup}`3`| explicit | Filter of the mipmap generator. |
| filmFilter | `struct`{sup}`3`| explicit | Filter of the image reconstruction filter. |

:::

:::{container} footer-block
:class: footer-block

\[1\]: `Linear`, `BVH` or `Hardware`

\[2\]: `ACES2065_1`, `ACES_CG`, `REC_709`, `REC_2020`, `DCI_P3` or`ADOBE_RGB`

\[3\]: A struct with 'radius' field (`float`) and a 'type' field (`string`) as either `Box`, `Tent`, `Gaussian`, `Mitchell-Netravali`. Radius unit is in pixels.

---
:::

`Linear` creates basically nothing, and intersections will be resolved by **linear search**. Only usefull for debugging purposes. `BVH` creates a hardware-agnostic LBVH accelerator, which is not competitive in terms of performance. It should be used when underlying hardware does not provide hardware-accelerated ray tracing capabilities. `Hardware` enables hardware-accelerated ray tracing and it should be used in general. For CPU Tracer, `Hardware` means running ray tracing via Embree which provides a sophisticated SIMD ray casting implementation.

### Render Config

Render config has two fields, `initialName` field (`string` type) which selects the renderer during initialization time. This field must match a valid renderer type name that is supported by a Tracer. Other field; the `Renderers` field, is a struct of structs each struct name inside this struct of structs must match renderer type name. Each struct's field is renderer parameter dependent. An example render config file is given below:

```{code-block} javascript
:name: rconfig_list
:caption: Render Config example

    // This renderer will be loaded
    // You can change this field and load different
    // renderer during load time.
    "initialName": "PathTracerRGB",

    "Renderers":
    {
        //=======================//
        //     Texture View      //
        //=======================//
        "TexView":
        {
            "totalSPP"    : 65536
        },
        //=======================//
        //   Surface Renderer    //
        //=======================//
        "Surface":
        {
            "totalSPP"             : 256,
            "doStochasticFilter"   : true,
            "tMaxAORatio" : 0.1,
            "renderType"  : "AmbientOcclusion"
        },
        //=======================//
        //    RGB Path Tracer    //
        //=======================//
        "PathTracerRGB":
        {
            "totalSPP"       : 32,
            "burstSize"      : 1,
            "renderMode"     : "Throughput",
            "sampleMode"     : "WithNEEAndMIS",
            "rrRange"        : [2, 20],
            "neeSamplerType" : "Uniform"
        }
    }
```

This approach ease development cycle and reduces file count when you implement/test different renderers. You can only change `initialName` field to switch renderers.

### Visor Config

Visor config has two mandatory fields that are called `VisorDLL` and `VisorOptions`.

:::{table} Visor DLL Struct
:widths: auto
:align: center

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| name      | `string` | explicit | Path to the .dll (.so) file in the system. OS specific lookup rules will apply. |
| construct | `string` | explicit | Mangled function name that constructs the tracer |
| destruct  | `string` | explicit | Mangled function name that destroys the tracer |

:::

:::{table} VisorOptions Struct
:widths: auto
:align: center

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| commandBufferSize | `int`| explicit | Static size of visor command buffer. Used to send commands to the tracer. |
| responseBufferSize | `int`| explicit | Static size of tracer response buffer. It is used by tracer to respond to the visor's commands. |
| enforceIGPU | `bool`| explicit | Force to use integrated GPU instead of first capable GPU. |
| displayHDR | `bool`| explicit | Try to tone map / display the tracer's output as HDR image. Ignored if current device (swap chain) is not HDR-capable. |
| windowSize | `vec2ui`| explicit | Initial window size of the visor. |
| realTime | `boolean` | explicit | Instead of letting the windowing system to trigger window draw event, continuously render window as fast as possible.

:::

---
