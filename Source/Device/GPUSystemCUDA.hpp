#pragma once

#include "GPUSystemCUDA.h"
#include "GPUSystem.h"

#include <nvtx3/nvToolsExt.h>

class NVTXKernelName;

class NVTXAnnotate
{
    private:
    nvtxDomainHandle_t  d;
    public:
                        NVTXAnnotate(const NVTXKernelName& kernelName);
                        NVTXAnnotate(const NVTXAnnotate&) = delete;
    NVTXAnnotate&       operator=(const NVTXAnnotate&) = delete;
                        ~NVTXAnnotate();
};

class NVTXKernelName
{
    friend class            NVTXAnnotate;
    nvtxDomainHandle_t      nvtxDomain;
    nvtxEventAttributes_t   eventAttrib = {0};
    public:
                            NVTXKernelName(nvtxDomainHandle_t domain,
                                           std::string_view name);
    NVTXAnnotate            Annotate() const;
};

inline NVTXAnnotate::NVTXAnnotate(const NVTXKernelName& kernelName)
    : d(kernelName.nvtxDomain)
{
    nvtxDomainRangePushEx(d, &(kernelName.eventAttrib));
}

inline NVTXAnnotate::~NVTXAnnotate()
{
    nvtxDomainRangePop(d);
}

inline NVTXKernelName::NVTXKernelName(nvtxDomainHandle_t domain,
                                      std::string_view name)
    : nvtxDomain(domain)
{
    eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    eventAttrib.message.registered = nvtxDomainRegisterStringA(nvtxDomain,
                                                               name.data());
}

inline NVTXAnnotate NVTXKernelName::Annotate() const
{
    return NVTXAnnotate(*this);
}

namespace CudaKernelCalls
{
    using namespace mray::cuda;

    template <class Lambda, uint32_t Bounds = StaticThreadPerBlock1D()>
    MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(Bounds)
    static void KernelCallLambdaCUDA(Lambda func)
    {
        func(KernelCallParamsCUDA());
    }
}

namespace mray::cuda
{
MRAY_GPU MRAY_CGPU_INLINE
KernelCallParamsCUDA::KernelCallParamsCUDA()
    : gridSize(gridDim.x)
    , blockSize(blockDim.x)
    , blockId(blockIdx.x)
    , threadId(threadIdx.x)
{}

template<auto Kernel, class... Args>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueKernel(std::string_view name,
                               KernelIssueParams p,
                               Args&&... fArgs) const
{
    #ifndef __CUDA_ARCH__
        static const NVTXKernelName kernelName = NVTXKernelName(nvtxDomain, name);
        NVTXAnnotate annotate = kernelName.Annotate();
    #endif;

    assert(p.workCount != 0);
    using namespace CudaKernelCalls;
    uint32_t blockCount = MathFunctions::DivideUp(p.workCount, StaticThreadPerBlock1D());
    uint32_t blockSize = StaticThreadPerBlock1D();

    Kernel<<<blockCount, blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Args>(fArgs)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Lambda>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueLambda(std::string_view name,
                               KernelIssueParams p,
                               //
                               Lambda&& func) const
{
    #ifndef __CUDA_ARCH__
        static const NVTXKernelName kernelName = NVTXKernelName(nvtxDomain, name);
        NVTXAnnotate annotate = kernelName.Annotate();
    #endif;

    assert(p.workCount != 0);
    static_assert(std::is_rvalue_reference_v<decltype(func)>,
                  "Not passing Lambda as rvalue_reference. This kernel call "
                  "would've been failed in runtime!");
    using namespace CudaKernelCalls;
    uint32_t blockCount = MathFunctions::DivideUp(p.workCount, StaticThreadPerBlock1D());
    uint32_t blockSize = StaticThreadPerBlock1D();

    KernelCallLambdaCUDA<Lambda>
    <<<blockCount, blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Lambda>(func)
    );
    CUDA_KERNEL_CHECK();
}

template<auto Kernel, class... Args>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueSaturatingKernel(std::string_view name,
                                         KernelIssueParams p,
                                         //
                                         Args&&... fArgs) const
{
    #ifndef __CUDA_ARCH__
        static const NVTXKernelName kernelName = NVTXKernelName(nvtxDomain, name);
        NVTXAnnotate annotate = kernelName.Annotate();
    #endif;

    assert(p.workCount != 0);
    using namespace CudaKernelCalls;

    const void* kernelPtr = static_cast<const void*>(Kernel);
    uint32_t threadCount = StaticThreadPerBlock1D();
    uint32_t blockCount = DetermineGridStrideBlock(kernelPtr,
                                                   p.sharedMemSize,
                                                   threadCount,
                                                   p.workCount);

    Kernel<<<blockCount, threadCount, p.sharedMemSize, stream>>>
    (
        std::forward<Args>(fArgs)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Lambda>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueSaturatingLambda(std::string_view name,
                                         KernelIssueParams p,
                                         //
                                         Lambda&& func) const
{
    #ifndef __CUDA_ARCH__
        static const NVTXKernelName kernelName = NVTXKernelName(nvtxDomain, name);
        NVTXAnnotate annotate = kernelName.Annotate();
    #endif;

    assert(p.workCount != 0);
    static_assert(std::is_rvalue_reference_v<decltype(func)>,
                  "Not passing Lambda as rvalue_reference. This kernel call "
                  "would've been failed in runtime!");
    using namespace CudaKernelCalls;
    const void* kernelPtr = static_cast<const void*>(&KernelCallLambdaCUDA<Lambda, StaticThreadPerBlock1D()>);
    uint32_t threadCount = StaticThreadPerBlock1D();
    uint32_t blockCount = DetermineGridStrideBlock(kernelPtr,
                                                   p.sharedMemSize,
                                                   threadCount,
                                                   p.workCount);


    KernelCallLambdaCUDA<Lambda>
    <<<blockCount, threadCount, p.sharedMemSize, stream>>>
    (
        std::forward<Lambda>(func)
    );
    CUDA_KERNEL_CHECK();
}

template<auto Kernel, class... Args>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueExactKernel(std::string_view name,
                                    KernelExactIssueParams p,
                                    //
                                    Args&&... fArgs) const
{
    #ifndef __CUDA_ARCH__
        static const NVTXKernelName kernelName = NVTXKernelName(nvtxDomain, name);
        NVTXAnnotate annotate = kernelName.Annotate();
    #endif;

    assert(p.gridSize != 0);
    using namespace CudaKernelCalls;
    Kernel<<<p.gridSize, p.blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Args>(fArgs)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Lambda, uint32_t Bounds>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueExactLambda(std::string_view name,
                                    KernelExactIssueParams p,
                                    //
                                    Lambda&& func) const
{
    #ifndef __CUDA_ARCH__
        static const NVTXKernelName kernelName = NVTXKernelName(nvtxDomain, name);
        NVTXAnnotate annotate = kernelName.Annotate();
    #endif;

    assert(p.gridSize != 0);
    static_assert(std::is_rvalue_reference_v<decltype(func)>,
                  "Not passing Lambda as rvalue_reference. This kernel call "
                  "would've been failed in runtime!");
    using namespace CudaKernelCalls;

    KernelCallLambdaCUDA<Lambda, Bounds>
    <<<p.gridSize, p.blockSize, p.sharedMemSize, stream>>>
    (
        std::forward<Lambda>(func)
    );
    CUDA_KERNEL_CHECK();
}

inline nvtxDomainHandle_t GPUQueueCUDA::ProfilerDomain() const
{
    return nvtxDomain;
}

}


