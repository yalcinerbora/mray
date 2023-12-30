#include "TextureCUDA.h"
#include "GPUTypes.h"

#include<memory>

//
//template<int D, class T>
//DeviceTextureCUDA<D, T>::DeviceTextureCUDA(const GPUDeviceCUDA& device,
//                       InterpolationType interp,
//                       EdgeResolveType eResolve,
//                       bool normalizeIntegers,
//                       bool normalizeCoordinates,
//                       bool convertSRGB,
//                       const TexDimType_t<D>& dim,
//                       uint32_t mipCount)
//    : TextureI<D>(dim, TextureChannelCount<T>::value, device, mipCount)
//    , interpType(interp)
//    , edgeResolveType(eResolve)
//    , normalizeIntegers(normalizeIntegers)
//    , normalizeCoordinates(normalizeCoordinates)
//    , convertSRGB(convertSRGB)
//{
//    cudaExtent extent = MakeCudaExtent<D>(this->dimensions);
//    cudaChannelFormatDesc d = cudaCreateChannelDesc<ChannelDescType_t<T>>();
//    CUDA_CHECK(cudaSetDevice(device->DeviceId()));
//    CUDA_MEMORY_CHECK(cudaMallocMipmappedArray(&data, &d, extent, mipCount));
//
//    // Allocation Done now generate texture
//    cudaResourceDesc rDesc = {};
//    cudaTextureDesc tDesc = {};
//
//    bool unormType = normalizeIntegers;
//
//    rDesc.resType = cudaResourceType::cudaResourceTypeMipmappedArray;
//    rDesc.res.mipmap.mipmap = data;
//
//    tDesc.addressMode[0] = DetermineAddressMode(eResolve);
//    tDesc.addressMode[1] = DetermineAddressMode(eResolve);
//    tDesc.addressMode[2] = DetermineAddressMode(eResolve);
//    tDesc.filterMode = DetermineFilterMode(interp);
//    tDesc.readMode = (unormType) ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
//
//    tDesc.sRGB = convertSRGB;
//    tDesc.borderColor[0] = 0.0f;
//    tDesc.borderColor[1] = 0.0f;
//    tDesc.borderColor[2] = 0.0f;
//    tDesc.borderColor[3] = 0.0f;
//    tDesc.normalizedCoords = normalizeCoordinates;
//    tDesc.mipmapFilterMode = DetermineFilterMode(interp);
//
//    tDesc.maxAnisotropy = 4;
//    tDesc.mipmapLevelBias = 0.0f;
//    tDesc.minMipmapLevelClamp = -100.0f;
//    tDesc.maxMipmapLevelClamp = 100.0f;
//
//    CUDA_CHECK(cudaCreateTextureObject(&this->texture, &rDesc, &tDesc, nullptr));
//}