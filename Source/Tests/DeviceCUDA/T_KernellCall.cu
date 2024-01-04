#include <gtest/gtest.h>

#include "Device/GPUSystem.h"
#include "Device/TextureCUDA.h"
#include "Device/GPUSystem.hpp"
#include "Core/Log.h"

#include "T_TextureTypes.h"

MRAY_GPU MRAY_CGPU_INLINE
void HelloWorldKernel(KernelCallParams params,
                      uint32_t& dRef,
                      size_t total)
{
    // A grid-stride loop
    for(uint32_t globalId = params.GlobalId(); globalId < total;
        globalId += params.TotalSize())
    {
        printf("Hello World tid:%u (%u)\n", globalId, dRef);
    }
}

// Cannot capture from the "TestBody"
// Warpping on a function
void KernelCallLambdaTester(const GPUSystem& system)
{
    const GPUQueue& queue = system.BestDevice().GetQueue(0);

    size_t launchSize = 4;
    auto LambdaKernel = [=] MRAY_GPU(KernelCallParams params)
    {
        // A grid-stride loop
        for(uint32_t globalId = params.GlobalId(); globalId < launchSize;
            globalId += params.TotalSize())
        {
            printf("Hello World Lambda tid:%d\n", globalId);
        }
    };

    queue.IssueKernelL
    (
        KernelIssueParams{static_cast<uint32_t>(launchSize), 0u},
        std::move(LambdaKernel)
    );

    queue.IssueSaturatingKernelL
    (
        KernelIssueParams{static_cast<uint32_t>(launchSize), 0u},
        std::move(LambdaKernel)
    );

    queue.IssueExactKernelL
    (
        KernelExactIssueParams{1u, 4u, 0u},
        std::move(LambdaKernel)
    );

}

TEST(GPUKernelCalls, FunctionPtr)
{
    GPUSystem system;
    const GPUQueue& queue = system.BestDevice().GetQueue(0);

    DeviceLocalMemory mem(system.BestDevice(), sizeof(uint32_t));

    uint32_t test = 999;
    Byte* dTestPtr = static_cast<Byte*>(mem);
    cudaMemcpy(dTestPtr, &test, sizeof(uint32_t),
               cudaMemcpyDefault);
    uint32_t& dRef123 = *reinterpret_cast<uint32_t*>(dTestPtr);

    // Free Function Call
    size_t size = 4;
    queue.IssueKernel<HelloWorldKernel>
    (
        KernelIssueParams{static_cast<uint32_t>(size), 0u},
        dRef123,
        4ull
    );

    queue.IssueSaturatingKernel<HelloWorldKernel>
    (
        KernelIssueParams{static_cast<uint32_t>(size), 0u},
        dRef123,
        4ull
    );

    queue.IssueExactKernel<HelloWorldKernel>
    (
        KernelExactIssueParams{1u, 4u, 0u},
        dRef123,
        4ull
    );

    // Lambda Call from a free function
    KernelCallLambdaTester(system);

    queue.Barrier().Wait();
}