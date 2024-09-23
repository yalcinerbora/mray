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
    const VulkanSystemView* handlesVk = nullptr;
    // Data related
    VulkanImage             hdrImage;
    VulkanImage             hdrSampleImage;
    VulkanImage             sdrImage;
    VulkanBuffer            outStageBuffer;
    VulkanDeviceMemory      stageMemory;
    VulkanDeviceMemory      imgMemory;
    //VulkanTimelineSemaphore
    Byte*                   hStagePtr       = nullptr;
    // Command related
    VulkanCommandBuffer     hdrCopyCommand;
    VulkanCommandBuffer     sdrCopyCommand;
    VulkanCommandBuffer     clearCommand;
    VulkanCommandBuffer     fullClearCommand;
    BS::thread_pool*        threadPool      = nullptr;
    ImageLoaderIPtr         imgLoader;
    //
    RenderImageInitInfo     initInfo;

    void                    GenerateSDRCopyCommand();
    void                    GenerateHDRCopyCommand();
    void                    GenerateClearCommand();
    void                    GenerateFullClearCommand();

    public:
    // Constructors & Destructor
                        RenderImagePool();
                        RenderImagePool(BS::thread_pool*,
                                        const VulkanSystemView&,
                                        const RenderImageInitInfo&);
                        RenderImagePool(const RenderImagePool&) = delete;
                        RenderImagePool(RenderImagePool&&) = default;
    RenderImagePool&    operator=(const RenderImagePool&) = delete;
    RenderImagePool&    operator=(RenderImagePool&&) = default;
                        ~RenderImagePool() = default;

    //
    void                SaveImage(IsHDRImage, const RenderImageSaveInfo& fileOutInfo,
                                  const VulkanTimelineSemaphore&);
    void                IssueClear(const VulkanTimelineSemaphore&);
    void                IssueFullClear(const VulkanTimelineSemaphore&);

    const VulkanImage&  GetHDRImage() const;
    const VulkanImage&  GetSampleImage() const;
    const VulkanImage&  GetSDRImage() const;
    size_t              UsedGPUMemBytes() const;
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