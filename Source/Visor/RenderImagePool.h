#pragma once

#include "VulkanTypes.h"
#include "ImageLoader/EntryPoint.h"
#include "Common/RenderImageStructs.h"

class FramePool;
namespace BS { class thread_pool; }

struct RenderImageInitInfo
{
    Vector2ui                       extent;
    MRayColorSpaceEnum              hdrColorSpace;
    Pair<MRayColorSpaceEnum, Float> sdrColorSpace;
    uint32_t                        depth           = 1;
    bool                            isSpectralPack  = false;
    Float                           sdrGamma        = Float(1.0);
};

class RenderImagePool
{
    public:
    enum IsHDRImage { HDR, SDR };

    private:
    // Data related
    VulkanImage             hdrImage;
    VulkanImage             hdrSampleImage;
    VulkanImage             sdrImage;
    VulkanBuffer            outStageBuffer;
    VulkanDeviceMemory      stageMemory;
    VulkanDeviceMemory      imgMemory;
    const VulkanSystemView* handlesVk       = nullptr;
    Byte*                   hStagePtr       = nullptr;
    // Command related
    VkCommandBuffer         hdrCopyCommand  = nullptr;
    VkCommandBuffer         sdrCopyCommand  = nullptr;
    VkCommandBuffer         clearCommand    = nullptr;
    BS::thread_pool*        threadPool      = nullptr;
    ImageLoaderIPtr         imgLoader;
    SemaphoreVariant        saveSemaphore   = {0, nullptr};
    SemaphoreVariant        clearSemaphore  = {0, nullptr};
    //
    RenderImageInitInfo     initInfo;

    void                    Clear();
    public:
    // Constructors & Destructor
                        RenderImagePool(BS::thread_pool*,
                                        const VulkanSystemView&);
                        RenderImagePool(BS::thread_pool*,
                                        const VulkanSystemView&,
                                        const RenderImageInitInfo&);
                        RenderImagePool(const RenderImagePool&) = delete;
                        RenderImagePool(RenderImagePool&&);
    RenderImagePool&    operator=(const RenderImagePool&) = delete;
    RenderImagePool&    operator=(RenderImagePool&&);
                        ~RenderImagePool();

    //
    SemaphoreVariant    SaveImage(SemaphoreVariant prevCmdSignal, IsHDRImage,
                                  const RenderImageSaveInfo& fileOutInfo);
    SemaphoreVariant    IssueClear(SemaphoreVariant prevCmdSignal);

    const VulkanImage&  GetHDRImage() const;
    const VulkanImage&  GetSampleImage() const;
    const VulkanImage&  GetSDRImage() const;
};

inline const VulkanImage& RenderImagePool::GetHDRImage() const
{
    return hdrImage;
}

inline const VulkanImage& RenderImagePool::GetSampleImage() const
{
    return hdrSampleImage;
}

inline const VulkanImage& RenderImagePool::GetSDRImage() const
{
    return sdrImage;
}