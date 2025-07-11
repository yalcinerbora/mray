#include <gtest/gtest.h>

#include "Device/GPUSystem.h"
#include "Device/GPUMemory.h"
#include "Device/GPUSystem.hpp"

#include <tuple>

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void HelloWorldKernel(Span<uint32_t> dOutBuffer,
                      const uint32_t& dReferenceValue,
                      uint32_t totalThreads)
{
    KernelCallParams params;
    // A grid-stride loop
    for(uint32_t globalId = params.GlobalId(); globalId < totalThreads;
        globalId += params.TotalSize())
    {
        dOutBuffer[globalId] = dReferenceValue + globalId;
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void NestedKernel(uint32_t* dOutBuffer,
                  const uint32_t& dReferenceValue,
                  uint32_t totalThreads)
{
    KernelCallParams params;
    // A grid-stride loop
    for(uint32_t globalId = params.GlobalId(); globalId < totalThreads;
        globalId += params.TotalSize())
    {
        dOutBuffer[globalId] = dReferenceValue + globalId;
    }
}

#ifndef MRAY_GPU_BACKEND_CPU

MRAY_KERNEL
void ParentKernel(uint32_t* dOutBuffer,
                  const uint32_t& dReferenceValue,
                  uint32_t totalThreads,
                  uint32_t deviceSMCount)
{
    KernelCallParams params;
    if(params.GlobalId() == 0)
    {
        const GPUQueue tailQueue(deviceSMCount, nullptr,
                                 DeviceQueueType::TAIL_LAUNCH);
        tailQueue.DeviceIssueWorkKernel<NestedKernel>
        (
            "GTest Nested Kernel",
            DeviceWorkIssueParams{totalThreads, 0},
            dOutBuffer,
            dReferenceValue,
            totalThreads
        );
    };
}

#endif

// TODO: Deduplicate these
void KernelCallFreeFunctionTester(const GPUSystem& system)
{
    static constexpr uint32_t HostInitValue = 999;
    uint32_t totalThreads = 32;
    uint32_t hReferenceTest = 512;
    const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);

    DeviceLocalMemory mem(system.BestDevice());
    Span<uint32_t> dWriteSpan;
    Span<uint32_t> dRefSpan;
    MemAlloc::AllocateMultiData(std::tie(dWriteSpan, dRefSpan),
                                mem,
                                {totalThreads, 1});
    // Copy reference to host
    Span<const uint32_t, 1> readSpan(&hReferenceTest, 1);
    system.Memcpy<uint32_t>(dRefSpan, readSpan);
    system.Memset(dWriteSpan, 0x12);

    // Construct reference type from host
    const uint32_t& dReference = dRefSpan.front();
    std::vector<uint32_t> hValues(totalThreads, HostInitValue);

    // Copy back to host and check
    auto CopyBackAndCheck = [&]()
    {
        queue.MemcpyAsync(Span<uint32_t>(hValues.begin(), hValues.end()),
                          ToConstSpan(dWriteSpan.subspan(0, totalThreads)));
        queue.Barrier().Wait();
        for(uint32_t i = 0; i < static_cast<uint32_t>(hValues.size()); i++)
        {
            EXPECT_EQ(hValues[i], hReferenceTest + i);
            hValues[i] = HostInitValue;
        }
        system.Memset(dWriteSpan, 0x00);
    };

    // ====================== //
    //       Work Kernel      //
    // ====================== //
    queue.IssueWorkKernel<HelloWorldKernel>
    (
        "GTest Hello World Kernel Work",
        DeviceWorkIssueParams{totalThreads, 0u},
        dWriteSpan,
        dReference,
        totalThreads
    );
    CopyBackAndCheck();

    // ====================== //
    //      Block Kernel      //
    // ====================== //
    queue.IssueBlockKernel<HelloWorldKernel>
    (
        "GTest Hello World Kernel Block",
        DeviceBlockIssueParams{1u, totalThreads, 0u},
        dWriteSpan,
        dReference,
        totalThreads
    );
    CopyBackAndCheck();
    queue.Barrier().Wait();
}

void KernelCallLambdaTester(const GPUSystem& system)
{
    static constexpr uint32_t HostInitValue = 999;
    uint32_t totalThreads = 32;
    uint32_t hReferenceTest = 512;
    const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);

    DeviceLocalMemory mem(system.BestDevice());
    Span<uint32_t> dWriteSpan;
    Span<uint32_t> dRefSpan;
    MemAlloc::AllocateMultiData(std::tie(dWriteSpan, dRefSpan),
                                mem,
                                {totalThreads, 1});
    // Copy reference to host
    Span<const uint32_t, 1> readSpan(&hReferenceTest, 1);
    system.Memcpy<uint32_t>(dRefSpan, readSpan);
    system.Memset(dWriteSpan, 0x12);

    // Construct pointer type from host
    const uint32_t* dPtr = dRefSpan.data();
    std::vector<uint32_t> hValues(totalThreads, HostInitValue);

    // Copy back to host and check
    auto CopyBackAndCheck = [&]()
    {
        queue.MemcpyAsync(Span<uint32_t>(hValues.begin(), hValues.end()),
                          ToConstSpan(dWriteSpan.subspan(0, totalThreads)));
        queue.Barrier().Wait();
        for(uint32_t i = 0; i < static_cast<uint32_t>(hValues.size()); i++)
        {
            EXPECT_EQ(hValues[i], hReferenceTest + i);
            hValues[i] = HostInitValue;
        }
        system.Memset(dWriteSpan, 0x00);
    };


    // Lambda Functor
    // We can copy everything except references
    auto LambdaKernel = [=] MRAY_GPU(KernelCallParams params)
    {
        // A grid-stride loop
        for(uint32_t globalId = params.GlobalId(); globalId < totalThreads;
            globalId += params.TotalSize())
        {
            dWriteSpan[globalId] = *dPtr + globalId;
        }
    };

    // ================ //
    //    Work Kernel   //
    // ================ //
    queue.IssueWorkLambda
    (
        "GTest Hello World Kernel Work",
        DeviceWorkIssueParams{totalThreads, 0u},
        std::move(LambdaKernel)
    );
    CopyBackAndCheck();


    // ====================== //
    //      Block Kernel      //
    // ====================== //
    queue.IssueBlockLambda
    (
        "GTest Hello World Kernel Block",
        DeviceBlockIssueParams{1u, totalThreads, 0u},
        std::move(LambdaKernel)
    );
    CopyBackAndCheck();

    queue.Barrier().Wait();
}

#ifndef MRAY_GPU_BACKEND_CPU

    void KernelCallNestedTester(const GPUSystem& system)
    {
        static constexpr uint32_t HostInitValue = 999;
        uint32_t totalThreads = 32;
        uint32_t hReferenceTest = 512;
        const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);

        DeviceLocalMemory mem(system.BestDevice());
        Span<uint32_t> dWriteSpan;
        Span<uint32_t> dRefSpan;
        MemAlloc::AllocateMultiData(std::tie(dWriteSpan, dRefSpan),
                                    mem,
                                    {totalThreads, 1});
        // Copy reference to host
        Span<const uint32_t, 1> readSpan(&hReferenceTest, 1);
        system.Memcpy<uint32_t>(dRefSpan, readSpan);
        system.Memset(dWriteSpan, 0x12);

        // Construct reference type from host
        const uint32_t& dReference = dRefSpan.front();
        std::vector<uint32_t> hValues(totalThreads, HostInitValue);

        // Copy back to host and check
        auto CopyBackAndCheck = [&]()
        {
            queue.MemcpyAsync(Span<uint32_t>(hValues.begin(), hValues.end()),
                              ToConstSpan(dWriteSpan.subspan(0, totalThreads)));
            queue.Barrier().Wait();
            for(uint32_t i = 0; i < static_cast<uint32_t>(hValues.size()); i++)
            {
                EXPECT_EQ(hValues[i], hReferenceTest + i);
                hValues[i] = HostInitValue;
            }
            system.Memset(dWriteSpan, 0x00);
        };


        // ====================== //
        //    Saturating Kernel   //
        // ====================== //
        queue.IssueWorkKernel<ParentKernel>
        (
            "GTest Parent Kernel Saturating",
            DeviceWorkIssueParams{totalThreads, 0u},
            dWriteSpan.data(),
            dReference,
            totalThreads,
            system.BestDevice().SMCount()
        );
        CopyBackAndCheck();

        // ====================== //
        //      Block Kernel      //
        // ====================== //
        queue.IssueBlockKernel<ParentKernel>
        (
            "GTest Parent Kernel Block",
            DeviceBlockIssueParams{1u, totalThreads, 0u},
            dWriteSpan.data(),
            dReference,
            totalThreads,
            system.BestDevice().SMCount()
        );
        CopyBackAndCheck();
        queue.Barrier().Wait();
    }

#endif

TEST(GPUKernelCalls, FreeFunction)
{
    GPUSystem system;

    // Free function kernel calls
    KernelCallFreeFunctionTester(system);
}

TEST(GPUKernelCalls, Lambda)
{
    GPUSystem system;
    // Lambda Call from a free function
    KernelCallLambdaTester(system);
}

#ifndef MRAY_GPU_BACKEND_CPU

    TEST(GPUKernelCalls, Nested)
    {
        GPUSystem system;
        KernelCallNestedTester(system);
    }

#endif