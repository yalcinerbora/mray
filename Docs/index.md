
# MRay Documentation

Welcome to the MRay Dcoumentation!

:::{important}

This documentation is nowhere near complete.

:::

**This Documentation Does**

- Test

**This Documentation Does Not:**

- Explain the code function by function basis (or class by class basis).

## Conventions

- Coordinate system is **right-handed** (i.e, direction of the `Math::Cross(a, b)`'s result will be determined by the right hand rule). Negative scaling of Z axis should be avoided.

- Up direction of the scene is the **Y** axis.

- Matrices are stored in **row-major** order. When multiplying vectors are assumed to be **X by 1** matrices (X rows, 1 column) where X is the length of the vector.

- Quaternions are stored as **`(w, x, y, z)`** (i.e., `Q[0]` will give the `w` component).

- `[0, 0]` (or `[0, 0, 0]`) coordinate of an image is at **bottom left**.

- Front face of a triangle is determined by **counter-clockwise (CCW)** order of the vertices.

- All texture's R (red) component is the least-significant value.  RGB components of a texture stored in increasing order.

- All 3-component textures are implicitly stored as 4-component textures (due to HW limitation).

-

:::{note}

If something is ambiguous, we are probably using OpenGL convention. You can assume as such.

:::

---
## Contents
```{toctree}
:caption: Getting Started
markdown/building.md
markdown/usage.md
markdown/glossary.md
```

```{toctree}
:caption: MRay Internals
markdown/projectLayout.md
markdown/tracer/tracer.md
```

```{toctree}
:caption: Scene Formats
markdown/scene/mrayScene.md
markdown/scene/usdScene.md
```



