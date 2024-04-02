#pragma once

#include "Device/GPUSystem.h"
#include "Core/TracerI.h"
#include "TracerTypes.h"
#include <map>

using GenericTextureView3D = Variant
<
    TextureView<3, Float>,
    TextureView<3, Vector2>,
    TextureView<3, Vector3>,
    TextureView<3, Vector4>
>;

using TextureView3DMap = std::map<TextureId, GenericTextureView3D>;