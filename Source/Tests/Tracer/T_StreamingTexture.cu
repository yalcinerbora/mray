#include <gtest/gtest.h>
#include "Tracer/StreamingTextureView.h"

//#include "Device/GPUSystem.hpp"

//MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
//void KCHashStatic(MRAY_GRID_CONSTANT const Span<uint64_t> out,
//                  MRAY_GRID_CONSTANT const SoASpanTest data)
//{
//    KernelCallParams kp;
//    for(uint32_t i = kp.GlobalId(); i < out.size();
//        i += kp.TotalSize())
//    {
//        uint32_t    a = data.Get<0>()[i];
//        uint64_t    b = data.Get<1>()[i];
//        Float       c = data.Get<2>()[i];
//        Vector2i    d = data.Get<3>()[i];
//
//        using RNGFunctions::HashPCG64::Hash;
//        uint64_t hash = 0;
//        for(uint32_t _ = 0; _ < 512; _++)
//        {
//            hash += Hash(a, b, c, d, hash);
//        }
//        out[i] = hash;
//    }
//}

TEST(StreamingTexture, TileCheck)
{
    using namespace StreamingTexParams;
    static_assert(PhysicalTileSize == 65536,
                  "Streaming Texture PhysicalTileSize must be 65536"
                  "for these test to work!");
}