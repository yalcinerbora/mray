#include <gtest/gtest.h>
#include <random>

#include "Device/GPUSystem.h"
#include "Device/TextureCUDA.h"
#include "Core/Log.h"

TEST(GPUTexture, TextureAlloc)
{
    GPUSystem system;


    TextureInitParams<1, Vector3f> tParams = {};
    tParams.eResolve = EdgeResolveType::WRAP;


    Texture<1, Vector3f> tex(system.BestDevice(), tParams);
}

