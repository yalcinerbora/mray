# MRay Scene Format

:::{note}
All of these definitions are in active development. These may change in future.
:::

MRay scene format is a simple scene format that only supports non-animated scenes (hopefully it will be added in future). The text format is `jasonc` (json with comment extensions) but trailing commas are not supported.

The design of the scene format follows declaration/definition style of implementation. Inside the file, there will be "Type-Group" declarations (for Primitives, Materials etc.) and Surface definitions (logical combination of these type groups).

## Type-Group Declarations

Scene file supports total of 7 type-groups which are `Cameras`, `Lights`, `Mediums`, `Transforms`, `Textures`, `Materials` and `Primitives`. All of these
lists must be present in a scene file. If any of the type-groups are not used, it's list then be defined as empty. An example type-group list is given below.

```{code-block} javascript
:name: td_list
:caption: Type-group deceleration list example, "Materials"

// An example list of Material Declarations
"Materials":
[
    // Struct that defines a single material
    {

        "id"    : 0,
        "type"  : "Lambert",
        "albedo": [0, 1, 0]     // Non-textured, perfectly green albedo
    },
    // Multiple materials on a single struct (useful for mentally grouping materials)
    {
        "id"    : [1, 2],       // Ids must be unique throughout the material definitions.
        "type"  : "Lambert"     // Type must be common in a struct
        "albedo":
        [
            {"texture": 0},     // If type is a struct, it must be a texture
            [0.5, 0.5, 0.5]
        ]
        "normal":               // Lambert material support optional normal map
        [
            "-",                // "-" means empty
            {"texture": 0}
        ]
    },
    ...
]
```
Each type-group list is a list of structs each struct defines one or more item.
Multiple structs can define multiple items. In example above, there are total of three `Lambert` materials (items) divided into 2 structs. Grouping the types into structs does not matter in MRay's eyes, it is only for the user to group the data. This list could've defined different types (i.e, a specular material).

:::{warning}
Only single type-group list (Primitives, Materials, Medium etc.) per type-group must be present on a single scene file.
:::

All structs in a type-group must define an unsigned integer `id` field (can be an array) and `type` field (must be a string). Internally, MRay will groups these according to their `type` field. `id` field is used to internally reference types inside the scene. Rest of the fields depend on the MRays implementation of the types (please check the <project:../tracer/tracer.md#default-tracer-types>).

You can define as many structs in these arrays as you want. Only the used structs will be loaded according to the [*surface definitions*](#definitions)

Below are types that does not exactly match the tracer's types or different parameters that are required for the system to work. For example; primitives can be defined inline according to their parameter. However; this is impractical for large triangle meshes. Instead, you can refer primitives from common file formats.

### Textures

Textures are the backbone of any rendering system. You can declare textures with the parameter's defined below.

:::{table} Texture fields and explanation
:widths: auto
:align: center

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| id             | `uint32_t`       | explicit      | As explained above, used as a reference throughout the scene |
| file           | `string`         | explicit      | Path of the texture. |
| channels       | `string`{sup}`1` | texture's channel count | Used channels of the texture. |
| asSigned       | `bool`           | false         | Do signed conversion to the value, given value is UNORM but it should be read as SNORM. |
| isColor        | `bool`           | true          | This texture value represents a color and must be color space converted. |
| interpolation  | `string`{sup}`2` | "Linear"      | Texture interpolation mode, applied to both across pixels and across mip levels. |
| edgeResolve    | `string`{sup}`3` | "Wrap"        | When a texture query is out-of-bounds, it is resolved via this parameter. |
| colorSpace     | `string`{sup}`4` | "Default"{sup}`5` | Overrides texture file's color space if available. |
| readMode       | `string`{sub}`6` | "Passthrough" | Simple read mode of the texture, useful for DDS textures, 2-channel normal maps |
| gamma          | `float`          | 1.0           | Overrides gamma value of the texture. |
| ignoreResClamp | `bool`           | false         | Ignore Tracer's tex clamp parameter. Useful for HDR boundary maps |
:::

:::{container} footer-block
:class: footer-block

\[1\]: `r`, `g`, `b`, `a`, `rg`, `rgb` or `rgba`

\[2\]: `Nearest` or `Linear`

\[3\]: `Wrap`, `Clamp` or `Mirror`

\[4\]: `ACES2065_1`, `ACES_CG`, `REC_709`, `REC_2020`, `DCI_P3` or `ADOBE_RGB`

\[5\]: If texture has a color space, it will be used. If not, the texture is assumed to be MRay's configured color space. A Warning will be generated during load time.

\[6\]: `Passthrough`, `Drop1`, `Drop2`, `Drop3`, `To3C_TsNormalBasic` or `To3C_TsNormalCoOcta`

---
:::

Channel parameter makes the loading system to load only these channels. Due to hardware limitations 3-channel textures may be padded as a 4-channel texture (which is the case of all desktop GPUs AFAIK.)

Texture may be automatically clamped to a certain resolution (modified via Tracer configuration parameter). This is an ad-hoc solution when GPU texture memory is not large enough for scene to fit. This mode may be undesired for certain textures (such as HDR maps). `ignoreResClamp` parameter overrides that behavior.

Scene loader will load mipmaps if given file contains them. If not mipmaps are automatically generated in initialization-time according to the Tracer's parameters.

**Supported Texture Formats:**
 - Through OIIO:
    - JPEG
    - PNG
    - TIFF
    - OpenEXR
    - DDS (If file's data is **not** block-compressed)
    - HDR

 - Internal:
    - DDS (If file's data is block-compressed)

:::{note}
MRay does not de-duplicate textures according to their `file` / `channel` parameters. If used, each texture with given `id` will be loaded into memory.
:::

### Transforms
Just like any other MRay type, transforms are also customizable. Most of the time given transform types would suffice. The transforms are specifically designed to be extensible for refining transforms of large instanced types (hair curves for example). Although only basic transform functionality is currently provided.

Below table is only for default types `Single` or `Multi`. If loader does not detect these, it will fallback to the common loading scheme and expects exact layout/name of the Type's attribute list. `Identity` is also supported then all attributes in the struct will be ignored

:::{table} Transform fields and explanation
:widths: auto
:align: center

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| id             | `uint32_t`       | explicit      | Id of the transform. |
| type           | `string`         | explicit      | Type of the transform. Must be either `Identity`, `Single` or `Multi` currently. |
| layout         | `string`{sup}`1` | explicit      | Structure layout of the transformation |
:::

:::{container} footer-block
:class: footer-block
\[1]\: `trs`, `matrix`

---
:::

Internally, transforms are stored as 4x4 transformation matrices. For simple scenes, user may want to explicitly define simple transform, rotate and scale sequences. To this end, user can use `trs` (transform, rotate and scale) parameter. `trs` will require `translate`, `rotate` and `scale`fields must be present on the struct. All these fields need to be a vector of 3 floats. `rotate` represents euler angles. These combined in an implicit order: `translate` * `rotZ` * `rotY` * `rotX` * `scale` * `T`, where `T` is the transformed type. (We do use OpenGL-style matrix calculations, meaning application order will start from `scale` and goes leftwards).

If `layout`filed is `matrix`, a `matrix` field must be present on the struct. `matrix` field should have series of 16 floats which are assumed to compose a row-major 4 by 4 matrix.

:::{note}
Multi-transform is currently under development and may not work properly.
:::

### Primitives
MRay scene format supports inline primitives and file-backed primitives. The term primitive is different from the internal primitive term. Scene `Primitive` term analogous to the `Primitive Batch` term of the MRay. So each primitive in the scene defines one or more MRay primitive.

:::{table} Primitive fields and explanation
:widths: auto
:align: center

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| id          | `uint32_t`       | explicit      | Id of the transform.  |
| type        | `string`         | explicit      | Type of the primitive |
| tag         | `string`{sup}`1` | explicit      | Tag field which will be used to determine the loading process. |
:::

:::{container} footer-block
:class: footer-block

\[1\]: `nodeSphere`, `nodeTriangle`, `nodeTriangleIndexed`, `assimp` or `gfg`

---
:::

:::{table} Extra fields when the tag is `assimp` or `gfg`
:widths: auto
:align: center
| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| file       | `string`   | explicit      | If primitive is file-backed, it will be fetched from this path. |
| innerIndex | `uint32_t` | explicit      | If file format supports multiple primitives, innerIndex{sup}`th` primitive will be loaded |
:::

:::{table} Extra fields when tag is `nodeTriangle` `nodeTriangleIndexed`
:widths: auto
:align: center
| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| index       | `vec3ui` | explicit  | If load mode is `nodeTriangleIndices`, this field represent vertex indices. Ignored if the `tag` is `nodeTriangle`  |
| position    | `vec3f`  | explicit  | vertex-position of the primitive(triangle mesh) |
| normal      | `vec3f`  | explicit  | vertex-normal |
| uv          | `vec2f`  | explicit  | |
:::

:::{table} Extra fields when tag is `nodeSphere`
:widths: auto
:align: center
| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| radius | `float` | explicit  | Sphere radius.        |
| center | `vec3f` | explicit  | Sphere center point.  |
:::

Unlike other types, primitive has slightly different meaning. Since most of the primitive data is implicitly batched (triangle mesh for example). Each primitive field must define batched primitives. For example, if you want to define a single triangle, you need to define it inside arrays.

**Supported File Formats:**
 - Through Assimp: [Please check assimp website.](https://the-asset-importer-lib-documentation.readthedocs.io/en/latest/about/introduction.html)
 - GFG (GFG is self developed simple file format)

### Materials & Mediums

In addition to their common responsibility, materials also define medium boundaries. Each material struct
optionally define `mediumFront` `mediumBack` fields (each refer to a medium via its `id`). If these fields
are not defined, both are assumed to be the boundary medium (see <project:#boundary>). To create an arbitrary volume field on the scene, a medium needs to be encapsulated via a primitive and a material (a surface). It is users responsibility to correctly define such fields so that medium information propagated correctly.

:::{error}
TODO: Add a figure to explain the statement above.
:::

### Lights

Lights can be either analytic or primitive-backed. For primitive-backed lights its `type` must be `Primitive` and `primitive` field must be present with a valid primitive id.

## Definitions

There are total of three definition arrays. These are `Surfaces`, `CameraSurfaces`
and `LightSurfaces`. Every surface defined in these arrays will be loaded and displayed.

### Boundary

Boundary is a special surface that must be defined in all scenes. It defines default `medium` (medium that
occupies every empty space between surfaces.), boundary `light` and its `transform`. Boundary light with
this specific transform will be triggered when a ray misses the scene.

### Surfaces

Surfaces are defined inside the `Surfaces` array. Each surface can be composed of multiple `primitive` / `material` pairs and and an optional `transform` field. If transform field is not present identity transform
is assumed.

Additionally, surface fields optionally define `cullFace` and `alphaMap` fields. `alphaMap` field can either
be `"-"` or refer to a valid texture that is a single-channel texture (i.e, its `channels` field must be either `r`, `g`, `b` or `a`).

#### Implicit Instancing

MRay natively supports implicit instancing. Given a `primitive` (or multiple primitives) field in a surface definition, MRay will create single accelerator (acceleration structure) if and only if the primitive type is
tagged as `'LOCALLY_CONSTANT_TRANSFORM'` inside the MRay itself (not in the scene definition). Each MRay primitive type supports either `'LOCALLY_CONSTANT_TRANSFORM'` or `'PER_PRIMITIVE_TRANSFORM'`.

As an example, the default `(P)Triangle` type is tagged with 'LOCALLY_CONSTANT_TRANSFORM' since it does not store any data to transform each primitive individually. However, `(P)TriangleSkinned` type represents a bone-hierarchy transformable triangles and it is tagged with 'PER_PRIMITIVE_TRANSFORM' (it requires per-vertex bone index and bone weight data).

A `(P)Triangle` id with different transforms will result in a single accelerator. However; when `(P)TriangleSkinned` primitive encountered with different transforms, multiple accelerators will
be generated for each surface definition. This is due to classic inverse transform application to the traversed ray can be applied for `(P)Triangle` but not for `(P)TriangleSkinned`.

:::{important}
Only single-level instancing is supported. You cannot encapsulate primitive / material pairs as a single surface
then use it inside another surface definition.
:::

#### Example Surface List
Synopsis of a surface definition list can be seen below.

```{code-block} javascript
:name: sd_list
:caption: Surface definition list example

// An example list of Surface definitions
"Surfaces":
[
    // Single tuple, single accelerator is generated for this transform / primitive pair.
    // Iff primitive type is tagged with 'LOCALLY_CONSTANT_TRANSFORM' (inside the MRay)
    {"transform": 0, "material": 5, "primitive": 0},
    // As stated above, this can be automatically instanced.
    {"transform": 1, "material": 5, "primitive": 0},
    // You can group primitives in to a single acceleration structure. Up to 8
    // primitives are supported in a single surface definition.
    {"transform": 0, "material": [5, 0, 1, 2], "primitive": [0, 1, 2, 3]},
    // This may be instanced as well (given primitive is tagged with 'LOCALLY_CONSTANT_TRANSFORM')
    {"transform": 3, "material": [11, 12, 13, 14], "primitive": [0, 1, 2, 3]},
    // If primitives should be two-sided, you need to explicitly state "cullFace" parameter
    // "cullFace" parameter must be define per primitive / material pair (i.e, check the alpha mapped
    // surface below).
    {"transform": 0, "material": 5, "primitive": 0, "cullFace": false},
    // If primitives require alpha maps you can define it with "alphaMap" parameter.
    // "-" represents empty.
    {"transform": 0, "material": [0, 1], "primitive": [11, 12], "alphaMap": ["-", {"texture": 2}]},
]
```

### Light Surfaces

Light surfaces more or less similar to the actual surfaces. However; they are in a separate field to do a renderer specific grouping (such as creating light sampler). Unlike surfaces, light surfaces do not support packed lights. Each light needs to have its own surface definition (`light` field must be an integer, not an array). Lights can also have `transform` field. Optionally, `medium` field can be defined when this light is inside a medium.

:::{note}
 Analytical Lights are "invisible" since MRay does not generate accelerator for them. For some renderers (i.e. a path tracer **without** NEE) these lights may not contribute to the scene's lighting.
:::

When a light is primitive-backed same surface instantiation rules apply to it. A primitive-backed light surface with same light id and different transform will be implicitly instantiated.

### Camera Surfaces

Camera surfaces are also separate and has the same semantics of light surfaces.

## Caveats

- You can not include other json files and compose scenes. It may be available in future versions.
- Admittedly, integer id system can be cumbersome for human editing. In future, we may define a name field (`string`) and expose a preprocess facility to hash these strings as ids.

## Example

Here is an example Cornell Box scene. Comments further explain certain capabilities and fallbacks of scene format.

```{code-block} javascript
:name: cornell_box_js
:caption: Cornell Box full scene definition in MRay format.

{
    // MRay is a simple scene format, and does not stored represent
    // scale information. Consistency of the values are user's responsibility.
    //
    //
    // Although, Declarations / Definitions are in order for this file, it is not required.
    "Cameras":
    [{
        "id"    : 0,
        "type"  : "Pinhole",
        "isFovX": true,
        "fov"   : 19.5,
        // Output image resolution is not defined in a scene file. It should match
        // the aspect ratio of the camera.
        "aspect": 1,
        "planes": [0.005, 90.0],

        // These initial transform values are on top of the given
        // camera surface definition's transform.
        // First, this transform is applied.
        "gaze"    : [0.0, 1.0, 0.0],
        "position": [0.0, 1.0, 6.8],
        "up"      : [0.0, 1.0, 0.0]
    }],

    "Lights" :
    [
        // Black Background
        // "Null" light is identity type of light (similar to identity transform).
        // It does not emit anything and does not geometry definitions.
        // This only makes sense to use as a boundary light.
        { "id" : 0, "type" : "Null"},
        // Ceiling Light
        {
            "id"       : 1,
            // Primitive-backed light
            "type"     : "Primitive",
            "primitive": 0,
            // Radiance value does not have a unit.
            // Radiance could've been texture-backed. In this case,
            // its syntax would be:
            // "radiance" : {"texture": 99}
            "radiance" : [68, 48, 16]
        }
    ],

    "Mediums" :
    [
        // Identity type Medium. It neither absorbs or scatter
        // light.
        { "id" : 0, "type" : "Vacuum"}
    ],

    "Transforms" :
    [
        // Identity type Transform. Internally MRay will optimize transformations
        // (i.e. the transformations will be noop instead of multiplying with an
        // identity matrix)
        {"id": 0, "type": "Identity"},
        //
        {
            // Here we define arrayed struct. This single struct defines 8
            // transformations at once.
            "id"    : [1, 2, 3, 4, 5, 7, 9, 10], // Ids must be unique, but it can be any 32-bit integer value
            // Type must be a single value throughout a struct.
            "type"  : "Single",
            // In this case layout is matrix, you can't mix-match layouts in a single struct
            "layout": "matrix",
            // Array size of this field must match array size of 'id' field.
            "matrix":
            [
                // Floor
                [
                    // Row-major ordering
                    2, 0,  0, 0,    // Row 1
                    0, 0,  2, 0,    // Row 2
                    0, -2, 0, 0,    // Row 3
                    0, 0,  0, 1     // Row 4
                ],
                // Ceiling
                [
                    2,  0,  0, 0,
                    0,  0, -2, 2,
                    0,  2,  0, 0,
                    0,  0,  0, 1
                ],
                // Left
                [
                    0, 0, 2, -1,
                    0, 2, 0,  1,
                    -2, 0, 0, 0,
                    0, 0, 0,  1
                ],
                // Right
                [
                    0, 0, -2, 1,
                    0, 2,  0, 1,
                    2, 0,  0, 0,
                    0, 0,  0, 1
                ],
                // Back
                [
                    2, 0, 0,  0,
                    0, 2, 0,  1,
                    0, 0, 2, -1,
                    0, 0, 0,  1
                ],
                // Front
                [
                    2, 0, 0,  0,
                    0, 2, 0,  1,
                    0, 0, -2, 1,
                    0, 0, 0,  1
                ],
                // Left Sphere
                [
                    1, 0, 0, -0.335439,
                    0, 1, 0, 0.3,
                    0, 0, 1, -0.291415,
                    0, 0, 0, 1
                ],
                // Right Sphere
                [
                    1, 0, 0, 0.328631,
                    0, 1, 0, 0.3,
                    0, 0, 1, 0.374592,
                    0, 0, 0, 1
                ]
            ]
        },
        {
            // Light
            "id"       : 6,
            "type"     : "Single",
            "layout"   : "trs",
            "translate": [-0.005, 1.999, -0.03],
            "rotate"   : [90, 0, 0],
            "scale"    : [0.47, 0.38, 1]
        }
    ],

    "Materials": [
    {
        // Diffuse Materials
        "id"    : [0, 1, 2],
        "type"  : "Lambert",
        "albedo":
        [
            [0.6300, 0.0650, 0.0500],   // Materials fields can be texture-backed too (similar syntax of 'radiance' field of a light)
            [0.14, 0.45, 0.091],
            [0.725, 0.71, 0.68]
        ]
    }],

    "Primitives": [
    {
        // Unit Plane
        // Inline primitive. In this case primitive is 'Triangle' it must be batched primitive.
        // So a single id represents two triangles.
        "id"      : 0,
        "type"    : "Triangle",
        "tag"     : "nodeTriangleIndexed",
        // The size of these fields can be arbitrarily large
        // and all of it will be loaded in to the memory.
        // So if 'index' field uses only subset of the data, there can
        // be a memory waste.
        //
        // If "tag" were "nodeTriangle", these fields would've been multiple of 3.
        // Each consecutive three vertices would have defined a triangle.
        "position":
        [
            [-0.5, -0.5, 0], [ 0.5, -0.5, 0],
            [ 0.5,  0.5, 0], [-0.5,  0.5, 0]
        ],
        "normal":
        [
            [0, 0, 1], [0, 0, 1],
            [0, 0, 1], [0, 0, 1]
        ],
        "uv":
        [
            [0, 0], [1, 0],
            [1, 1], [0, 1]
        ],
        // Since  this is indexed triangle, this field must be multiple of three (more specifically
        // it must be array of vector of three integers).
        "index": [[0, 1, 2], [0, 2, 3]]
    },
    {
        // Sphere
        "id"    : 1,
        "type"  : "Sphere",
        "tag"   : "nodeSphere",
        "radius": 0.3,
        "center": [0, 0.0, 0]
    }],

    // Cornell Box scene does not use any textures, however this field must be present in the file.
    "Textures": [],

    //================================================//
    //        Actual Scene Related Declarations       //
    //================================================//
    "Boundary" :
    {
        "medium"    : 0,
        "light"     : 0,
        "transform" : 0
    },

    "Surfaces":
    [
        // Floor
        { "transform":  1, "material": 2, "primitive": 0},  // Implicitly back face culling is on and
                                                            // alpha map is not present on this surface.
        // Ceiling
        { "transform":  2, "material": 2, "primitive": 0},
        // Back Wall
        { "transform":  5, "material": 2, "primitive": 0},
        // Left Wall
        { "transform":  3, "material": 0, "primitive": 0},
        // Right Wall
        { "transform":  4, "material": 1, "primitive": 0},
        // Front Wall
        { "transform":  7, "material": 2, "primitive": 0},
        // Spheres
        { "transform":  9, "material": 3, "primitive": 1},
        { "transform": 10, "material": 4, "primitive": 1}
    ],

    "LightSurfaces": [{"light": 1, "transform": 6}], // Light has special transform but its
                                                     // medium is implicitly "Boundary.medium"
                                                     // since it is not explicitly defined.
    "CameraSurfaces":[{"camera": 0}]                 // Camera does not define its transform,
                                                     // so it is identity matrix.
}

```