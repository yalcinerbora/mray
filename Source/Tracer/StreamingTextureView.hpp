#pragma once

#include "StreamingTextureView.h"

MR_GF_DEF
Vector4 TextureStreamingContext::FetchTextureBase(VirtualTextureId texId, Vector2 uv)
{
    return FetchTextureLod(texId, uv, Float(0));
}

MR_GF_DEF
Vector4 TextureStreamingContext::FetchTextureGrad(VirtualTextureId tid, Vector2 uv,
                                                  Vector2 dpdx, Vector2 dpdy)
{
    uint32_t packIndex = tid.PackIndex();
    uint32_t virtTexIndex = tid.TextureIndex();
    VirtualTexInfo texInfo = texInfoList[virtTexIndex];
    // Find the virtual tile
    auto [virtMip, virtTile] = GenVirtualMipAndTileGrad(texInfo, uv, dpdx, dpdy).AsArray();
    // HT Lookup
    uint64_t key = GenerateVirtualTextureKey(virtTexIndex, virtMip, virtTile);
    Optional<PhysicalTileId> physicalTileOpt = tileLookup[packIndex].Search(key);
    if(!physicalTileOpt.has_value())
    {
        uint32_t tileBitOffset = TileBitOffset(texInfo, virtMip, virtTile);
        // Request the tile by setting it bit to one
        requestBits.SetBitParallel(tileBitOffset, true);
        return Vector4(BIG_CYAN(), 0);
    }
    // Calculate the fetch parameters
    PhysicalTileId physicalTilePack = physicalTileOpt.value();
    uint32_t physicalTexIndex = physicalTilePack.ArrayIndex();

    auto [uvP, dpdxP, dpdyP] = CalculatePhysicalUVsGrad(texInfo,
                                                        physicalTilePack,
                                                        uv, dpdx, dpdy);
    // Runtime find the texture
    uint32_t channels = FetchChannelCount(texInfo.pixType);
    uint32_t typeIndex = tid.TypeIndex();
    switch(channels)
    {
        case 1:
        {
            TextureView<2, Float> tex = textureViews.texViews1C[typeIndex][physicalTexIndex];
            return Vector4(tex(uvP, dpdxP, dpdyP), 0, 0, 0);
        }
        case 2:
        {
            TextureView<2, Vector2> tex = textureViews.texViews2C[typeIndex][physicalTexIndex];
            return Vector4(tex(uvP, dpdxP, dpdyP), 0, 0);
        }
        case 4:
        {
            TextureView<2, Vector4> tex = textureViews.texViews4C[typeIndex][physicalTexIndex];
            return tex(uvP, dpdxP, dpdyP);
        }
    }
    return Vector4(BIG_CYAN(), 0);
}

MR_GF_DEF
Vector4 TextureStreamingContext::FetchTextureLod(VirtualTextureId tid, Vector2 uv,
                                                 Float lod)
{
    uint32_t packIndex = tid.PackIndex();
    uint32_t virtTexIndex = tid.TextureIndex();
    VirtualTexInfo texInfo = texInfoList[virtTexIndex];
    // Find the virtual tile
    auto [virtMip, virtTile] = GenVirtualMipAndTileLod(texInfo, uv, lod).AsArray();
    // HT Lookup
    uint64_t key = GenerateVirtualTextureKey(virtTexIndex, virtMip, virtTile);
    Optional<PhysicalTileId> physicalTileOpt = tileLookup[packIndex].Search(key);
    if(!physicalTileOpt.has_value())
    {
        uint32_t tileBitOffset = TileBitOffset(texInfo, virtMip, virtTile);
        // Request the tile by setting it bit to one
        requestBits.SetBitParallel(tileBitOffset, true);
        return Vector4(BIG_CYAN(), 0);
    }
    // Calculate the fetch parameters
    PhysicalTileId physicalTilePack = physicalTileOpt.value();
    uint32_t physicalTexIndex = physicalTilePack.ArrayIndex();

    auto [uvP, lodP] = CalculatePhysicalUVsLod(texInfo,
                                               physicalTilePack,
                                               uv, lod);
    // Runtime find the texture
    uint32_t channels = FetchChannelCount(texInfo.pixType);
    uint32_t typeIndex = tid.TypeIndex();
    switch(channels)
    {
        case 1:
        {
            TextureView<2, Float> tex = textureViews.texViews1C[typeIndex][physicalTexIndex];
            return Vector4(tex(uvP, lodP), 0, 0, 0);
        }
        case 2:
        {
            TextureView<2, Vector2> tex = textureViews.texViews2C[typeIndex][physicalTexIndex];
            return Vector4(tex(uvP, lodP), 0, 0);
        }
        case 4:
        {
            TextureView<2, Vector4> tex = textureViews.texViews4C[typeIndex][physicalTexIndex];
            return tex(uvP, lodP);
        }
    }
    return Vector4(BIG_CYAN(), 0);
}

MR_GF_DEF
Vector4 StreamingTextureView::operator()(UV uv) const
{
    return context->FetchTextureBase(texId, uv);
}

MR_GF_DEF
Vector4 StreamingTextureView::operator()(UV uv, UV dpdx, UV dpdy) const
{
    return context->FetchTextureGrad(texId, uv, dpdx, dpdy);
}

MR_GF_DEF
Vector4 StreamingTextureView::operator()(UV uv, Float lod) const
{
    return context->FetchTextureLod(texId, uv, lod);
}