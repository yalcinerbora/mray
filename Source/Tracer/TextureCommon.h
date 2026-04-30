#pragma once

#include "Core/BitFunctions.h"
#include "Core/TracerConstants.h"
#include <array>

using TexMipBitSet = Bitset<TracerConstants::MaxTextureMipCount>;

template<class T>
using MipArray = Array<T, TracerConstants::MaxTextureMipCount>;
