#pragma once

#include "VulkanTypes.h"
#include "ImageLoader/EntryPoint.h"
#include "CommonHeaders/RenderImageStructs.h"

class FramePool;
namespace BS { class thread_pool; }

struct RenderImageInitInfo
{
    Vector2i                        extent;
    MRayColorSpaceEnum              hdrColorSpace;
    Pair<MRayColorSpaceEnum, Float> sdrColorSpace;
    uint32_t depth      = 1;
    bool isSpectralPack = false;
    Float sdrGamma      = Float(1.0);
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
    VkDeviceMemory          stageMemory     = nullptr;
    VkDeviceMemory          imgMemory       = nullptr;
    const VulkanSystemView* handlesVk       = nullptr;
    Byte*                   hStagePtr       = nullptr;
    // Command related
    VkCommandBuffer         hdrCopyCommand  = nullptr;
    VkCommandBuffer         sdrCopyCommand  = nullptr;
    BS::thread_pool*        threadPool      = nullptr;
    ImageLoaderIPtr         imgLoader;
    VkSemaphore             saveSemaphore   = nullptr;
    uint64_t                semCounter      = 0;
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
    void                SaveImage(VkSemaphore prevCmdSignal, IsHDRImage,
                                  const RenderImageSaveInfo& fileOutInfo);
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