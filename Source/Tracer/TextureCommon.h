#pragma once

#include "Core/BitFunctions.h"
#include "Core/TracerI.h"
#include <array>

using TexMipBitSet = Bitset<TracerConstants::MaxTextureMipCount>;

template<class T>
using MipArray = std::array<T, TracerConstants::MaxTextureMipCount>;
