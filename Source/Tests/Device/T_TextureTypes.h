#pragma once

#include <gtest/gtest.h>

#include "Device/GPUTypes.h"

template <uint32_t DIM, class TT>
struct DTPair
{
    static constexpr auto D = DIM;
    using T = TT;
};

template <class T>
class GPUTextureTest : public testing::Test
{
    public:
    static constexpr auto D = T::D;
    using ChannelType = typename T::T;
    using ParamType = TextureInitParams<D>;
    using SizeType = TextureExtent<D>;
};

using Implementations = ::testing::Types
<
    DTPair<1, Float>,
    DTPair<1, Vector2>,
    DTPair<1, Vector3>,
    DTPair<1, Vector4>,
    DTPair<1, uint8_t>,
    DTPair<1, Vector2uc>,
    DTPair<1, Vector3uc>,
    DTPair<1, Vector4uc>,
    DTPair<1, int8_t>,
    DTPair<1, Vector2c>,
    DTPair<1, Vector3c>,
    DTPair<1, Vector4c>,
    DTPair<1, uint16_t>,
    DTPair<1, Vector2us>,
    DTPair<1, Vector3us>,
    DTPair<1, Vector4us>,
    DTPair<1, int16_t>,
    DTPair<1, Vector2s>,
    DTPair<1, Vector3s>,
    DTPair<1, Vector4s>,

    DTPair<2, Float>,
    DTPair<2, Vector2>,
    DTPair<2, Vector3>,
    DTPair<2, Vector4>,
    DTPair<2, uint8_t>,
    DTPair<2, Vector2uc>,
    DTPair<2, Vector3uc>,
    DTPair<2, Vector4uc>,
    DTPair<2, int8_t>,
    DTPair<2, Vector2c>,
    DTPair<2, Vector3c>,
    DTPair<2, Vector4c>,
    DTPair<2, uint16_t>,
    DTPair<2, Vector2us>,
    DTPair<2, Vector3us>,
    DTPair<2, Vector4us>,
    DTPair<2, int16_t>,
    DTPair<2, Vector2s>,
    DTPair<2, Vector3s>,
    DTPair<2, Vector4s>,

    DTPair<3, Float>,
    DTPair<3, Vector2>,
    DTPair<3, Vector3>,
    DTPair<3, Vector4>,
    DTPair<3, uint8_t>,
    DTPair<3, Vector2uc>,
    DTPair<3, Vector3uc>,
    DTPair<3, Vector4uc>,
    DTPair<3, int8_t>,
    DTPair<3, Vector2c>,
    DTPair<3, Vector3c>,
    DTPair<3, Vector4c>,
    DTPair<3, uint16_t>,
    DTPair<3, Vector2us>,
    DTPair<3, Vector3us>,
    DTPair<3, Vector4us>,
    DTPair<3, int16_t>,
    DTPair<3, Vector2s>,
    DTPair<3, Vector3s>,
    DTPair<3, Vector4s>
>;

TYPED_TEST_SUITE(GPUTextureTest, Implementations);
