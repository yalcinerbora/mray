#pragma once

#include "GPUSystemCPU.h"
#include "../GPUSystem.h"

static constexpr uint32_t WarpSize()
{
    return 1u;
}

template<uint32_t LOGICAL_WARP_SIZE = WarpSize()>
MRAY_GPU MRAY_GPU_INLINE
static void WarpSynchronize()
{
}

MRAY_GPU MRAY_GPU_INLINE
static void BlockSynchronize()
{
}

MRAY_GPU MRAY_GPU_INLINE
static void ThreadFenceGrid()
{
}

namespace mray::host
{

MRAY_GPU MRAY_CGPU_INLINE
KernelCallParamsCPU::KernelCallParamsCPU()
    : gridSize(0)
    , blockSize(0)
    , blockId(0)
    , threadId(0)
{}

template<auto Kernel, class... Args>
MRAY_HOST inline
void GPUQueueCPU::IssueKernel(std::string_view name,
                              KernelIssueParams p,
                              Args&&... fArgs) const
{

}

template<class Lambda>
MRAY_HOST inline
void GPUQueueCPU::IssueLambda(std::string_view name,
                              KernelIssueParams p,
                              //
                              Lambda&& func) const
{

}

template<auto Kernel, class... Args>
MRAY_HOST inline
void GPUQueueCPU::IssueSaturatingKernel(std::string_view name,
                                        KernelIssueParams p,
                                        //
                                        Args&&... fArgs) const
{

}

template<class Lambda>
MRAY_HOST inline
void GPUQueueCPU::IssueSaturatingLambda(std::string_view name,
                                        KernelIssueParams p,
                                        //
                                        Lambda&& func) const
{

}

template<auto Kernel, class... Args>
MRAY_HOST inline
void GPUQueueCPU::IssueExactKernel(std::string_view name,
                                   KernelExactIssueParams p,
                                   //
                                   Args&&... fArgs) const
{

}

template<class Lambda, uint32_t Bounds>
MRAY_HOST inline
void GPUQueueCPU::IssueExactLambda(std::string_view name,
                                   KernelExactIssueParams p,
                                   //
                                   Lambda&& func) const
{

}

template<auto Kernel, class... Args>
MRAY_GPU inline
void GPUQueueCPU::DeviceIssueKernel(std::string_view name,
                                    KernelIssueParams p,
                                    Args&&... fArgs) const
{

}

template<class Lambda>
MRAY_GPU inline
void GPUQueueCPU::DeviceIssueLambda(std::string_view name,
                                    KernelIssueParams p,
                                    //
                                    Lambda&& func) const
{

}

template<auto Kernel, class... Args>
MRAY_GPU inline
void GPUQueueCPU::DeviceIssueSaturatingKernel(std::string_view name,
                                              KernelIssueParams p,
                                              //
                                              Args&&... fArgs) const
{

}

template<class Lambda>
MRAY_GPU inline
void GPUQueueCPU::DeviceIssueSaturatingLambda(std::string_view name,
                                              KernelIssueParams p,
                                              //
                                              Lambda&& func) const
{

}

template<auto Kernel, class... Args>
MRAY_GPU inline
void GPUQueueCPU::DeviceIssueExactKernel(std::string_view name,
                                         KernelExactIssueParams p,
                                         //
                                         Args&&... fArgs) const
{

}

template<class Lambda, uint32_t Bounds>
MRAY_GPU inline
void GPUQueueCPU::DeviceIssueExactLambda(std::string_view name,
                                         KernelExactIssueParams p,
                                         //
                                         Lambda&& func) const
{

}

}


