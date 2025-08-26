#pragma once

#include "Device/GPUTexture.h"
#include "Device/GPUTextureView.h"

#include "Core/BitFunctions.h"
#include "Core/Types.h"
#include "Core/TracerI.h"
#include "Core/Variant.h"

#include <array>

using TexMipBitSet = Bitset<TracerConstants::MaxTextureMipCount>;

template<class T>
using MipArray = std::array<T, TracerConstants::MaxTextureMipCount>;
