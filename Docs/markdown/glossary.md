# Glossary

What is what?
MRay has very common terminologies however I think it is better to list these somewhere to express

:::{table} Glossary (Generic)
:widths: auto
:align: center

| Concept | Description |
| ------- | ----------- |
| MRay             | The executable name and project name. |
| Tracer           | The name of the backend. Name comes from ray tracing. It is prefixed with the device/platform name. (i.e., `TracerDLL_CPU` or `TracerDLL_CUDA`) |
| Host/CPU          | The hardware and runtime (in this case the runtime is the OS) that contains the Device(s). |
| Device/GPU        | Self explanatory.  |
| Visor             | Frontend of the MRay. It is a Vulkan-based image display and GUI. |
| Renderer          | Renders the image according to a specific algorithm. A tracer can have multiple renderers. Only one renderer can be run at the same time. |
| XXGroup          | Since GPU hardware is massively parallel, every type is stored in a 'Group'. `XX` can be `Accelerator`, `Medium`, `Transform`, `Primitive`, `Camera` or `Light`. There is only a *single* group for each `XX` type. |
| Accelerator       | Similar to DXR (DirectX Ray tracing) api, Accelerator is an abstraction that accelerates ray geometry intersections. Most probably, it is implemented as Bounding Volume Hierarchies (BVHs) |
| Base Accelerator  | Base accelerator is top-level accelerator structure that groups accelerators (indirectly geometries) and enables instancing. |
| Texture           | If you have done any graphics programming you probably know the term textures. In MRays terms, it is exactly the same. It is 1D, 2D or 3D spatial data that can be accessed with indices or normalized values. It has automatic access of data in-between of the discretized values via interpolation. Additionally, it has explicit modes for out-of-bounds access. |
| Medium            | Aka. 'participating media'. Discretization of a volume that contains matter. Geometry of the volume is not defined by a medium. But matter density and light transport-related data is stored by this abstraction. |
| Camera            | In fancier terms 'Sensor'. Logically the opposite of 'Light'. It absorbs light over a 2D canvas (film) and produces the output image. |
| Light             | Provides illumination to the scene. Lights both contain geometry (either analytically or via a primitive). Although an arbitrary material can also emit light, lights may be treated differently depending on the renderer (i.e., NEE). |
| MetaLight         | Tagged union of light types, a renderer can use the list of these to use better algorithms (NEE). |
| Transform         | Contains information about transformation. May require information from a Primitive to transform normals rays etc.. |
| Transform Context | Collaboration of a Transform and a Primitive. the Primitive may or may not contribute to the Transform Context. This abstraction is used to transform rays/normals etc. |
| Primitive         | Unit of geometry (i.e, a single triangle is a primitive). Occupies space in a scene. |
| Primitive Batch   | Group of primitives (i.e, triangle mesh is a primitive batch). Only a single material can only act on a Primitive Batch. |
| Material          | Responsible containing BxDF information (sampling routines and data). May contain multiple BxDFs.|
| Surface           | Patch of a transformed Primitive. It has position/normal data etc., as well as the rate of change (aka. differentials) of those data. |
| XX Key            | MRay utilizes tagged pointer-like system ([see PBR Tagged Pointers](https://www.pbr-book.org/4ed/Utilities/Containers_and_Memory_Management#TaggedPointers)) however; it is quite different. Each type-group (`Primitive`, `Transform`, `Material` etc.) has a key with variable sized upper and lower fields. Upper field represents the actual type (i.e, for transform's case, `Identity` and `Single` etc.) and lower field represents the actual object. Polymorphism handled in host level, Each kernel is instantiated with different types and these two fields of the keys are used to call kernels / access data. `XX` can be `Light`, `Camera`, `Primitive`, `Accelerator`, `Transform` or `Medium`. If this system is not capable to devise an algorithm, we fallback to tagged unions (such as 'MetaLight'). |

:::

These terms below come up during GPU-specific algorithm design, provided here to aid reading the source code. All in all, we use NVIDIA CUDA terms for most of the time.

:::{table} Glossary (GPU-specific Terms)
:widths: auto
:align: center

| Concept | Description |
| ------- | ----------- |
| Kernel          | C++ function that is run on the GPU. Control flow is implicitly parallel. |
| Grid            | Total number of blocks in a kernel launch.  |
| Block           | Logical partitioning of the threads (and implicitly warps). Can have shared memory. (Vulkan/GLSL: `work group`)  |
| Warp            | Group of thread that are executed in parallel. Since forever, warp size of NVIDIA GPUs is 32 (\# of thread in a warp). Special instructions may apply to these depending on the HW. (Vulkan/GLSL: `sub-group`) |
| Thread          | Leaf unit of the GPU parallel execution. This term is completely different (please check the paragraph below). (Vulkan/GLSL: `invocation`)|
| Shared Memory   | Dynamically or statically allocated memory for each block. Threads in a block can access this and share information.
| XX-Stride Loop  | Where `XX` either `Grid`, `Block` or `Warp`. Each `XX` is dedicated to multiple 'data' instead of exactly single 'data'. This pattern can be useful when you do not want to fully utilize entire GPU with a single Kernel but you want multiple kernels to run in parallel. More detailed description can be found [here](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/). |
| Streaming Multiprocessor (SM)  | The 'core' of the GPU (this should not be confused with the NVIDIA term CUDA core). During kernel execution, each SM acquires one or more blocks and executes them. To hide latency, SM juggles warps. SM is not exactly the HW design since Ampere-class GPUs AFAIK but it is still useful mental model of GPU execution. |
| Queue            | Unlike other terms, this term is different compared to CUDA. In CUDA this is called `stream`. Queues are used to send Kernels to the GPU. In Vulkan terms, this is 'compute queue' however; with an important difference. In Vulkan, works are issued to a queue in order but does **not necessarily completed in order**. But for CUDA streams (hence MRay Queues), every Kernel is guaranteed to be completed in order *and whatever side effect they introduce is visible* to the next Kernel in the queue. |

:::

GPU execution is complex and contains both parallel execution and concurrent execution. Threads in a warp is executed in parallel, but block-local warps are executed concurrently (or sometimes parallel depending on the device{sup}`1`). Again, adjacent blocks may be executed concurrently (if they reside on the same SM) but overall blocks are executed in parallel. Depending on the issue pattern, kernels would also be executed in parallel. When designing an algorithm this execution pattern should be taken in consideration.


:::{container} footer-block

\[1\]: Old GPUs had exactly 32 execution units (more or less ALUs) but modern GPUs may have more than that. So SM scheduler can issue warps in parallel.

---
:::

:::{note}
Unlike CUDA, we only support 1D grids for simplicity.
:::




