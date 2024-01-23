#include "Distributions.h"
#include "Core/MemAlloc.h"

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgorithms.h"

#include <algorithm>
#include <numeric>

//#ifdef MRAY_GPU_BACKEND_CUDA
//    #include <cub/block/block_scan.cuh>
//    #include <cub/block/block_load.cuh>
//    #include <cub/block/block_store.cuh>
//#endif


namespace Distributions
{

static constexpr uint32_t TPB = StaticThreadPerBlock1D();

#ifdef MRAY_GPU_BACKEND_CUDA

    MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
    void KCCopyScanY(Span<Float> dYCDFs,
                     // I-O
                     Span<const Float> dXCDFs)
    {
        KernelCallParams kp;
        if(kp.blockId != 0) return;

        Float aggregate = 0;
        auto PrefixLoader = [&](Float iterationAggregate)
        {
            Float temp = aggregate;
            aggregate += iterationAggregate;
            return temp;
        };

        assert(dXCDFs.size() % dYCDFs.size() == 0);
        static constexpr uint32_t ITEMS_PER_THREAD = 4;
        static constexpr uint32_t DATA_PER_BLOCK = TPB * ITEMS_PER_THREAD;

        uint32_t yCount = static_cast<uint32_t>(dYCDFs.size());
        uint32_t xCount = static_cast<uint32_t>(dXCDFs.size() / yCount);

        using BlockStore = cub::BlockStore<Float, TPB, ITEMS_PER_THREAD, cub::BLOCK_STORE_VECTORIZE>;
        using BlockScan = cub::BlockScan<Float, TPB>;

        uint32_t processedItemsSoFar = Float(0);
        while(processedItemsSoFar != yCount)
        {
            uint32_t validItems = min(DATA_PER_BLOCK, yCount - processedItemsSoFar);
            auto dSubBlockOut = dYCDFs.subspan(processedItemsSoFar, validItems);

            // Awfully strided mem read
            Float dataRegisters[ITEMS_PER_THREAD];
            UNROLL_LOOP
            for(uint32_t i = 0; i < ITEMS_PER_THREAD; i++)
            {
                // Contiguous index
                uint32_t index = processedItemsSoFar + kp.threadId * ITEMS_PER_THREAD + i;
                uint32_t indexStrided = index * xCount + (xCount - 1);
                dataRegisters[i] = (index >= yCount) ? Float(0) : dXCDFs[indexStrided];
            }

            // Scan
            BlockScan().InclusiveScan(dataRegisters, dataRegisters,
                                      [](Float a, Float b) {return a + b; },
                                      PrefixLoader);

            // Store
            if(validItems == DATA_PER_BLOCK) [[likely]]
                BlockStore().Store(dSubBlockOut.data(), dataRegisters);
            else
                BlockStore().Store(dSubBlockOut.data(), dataRegisters, validItems);


            processedItemsSoFar += validItems;
            MRAY_DEVICE_BLOCK_SYNC();
        }
    };

    MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
    void KCNormalizeXY(Span<Float> dXCDFs, Span<Float> dYCDFs)
    {
        KernelCallParams kp;
        static constexpr uint32_t ITEMS_PER_THREAD = 4;
        static constexpr uint32_t DATA_PER_BLOCK = TPB * ITEMS_PER_THREAD;

        using BlockLoad = cub::BlockLoad<Float, TPB, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
        using BlockStore = cub::BlockStore<Float, TPB, ITEMS_PER_THREAD, cub::BLOCK_STORE_VECTORIZE>;

        MRAY_SHARED_MEMORY Float sTotalRecip;

        auto NormalizeRow = [&](Span<Float> rowData)
        {
            uint32_t rowCount = static_cast<uint32_t>(rowData.size());

            if(kp.threadId == 0) sTotalRecip = Float(1) / rowData[rowCount - 1];
            MRAY_DEVICE_BLOCK_SYNC();

            uint32_t processedItemsSoFar = 0;
            while(processedItemsSoFar != rowCount)
            {
                uint32_t validItems = min(DATA_PER_BLOCK, rowCount - processedItemsSoFar);
                auto dSubBlockInOut = rowData.subspan(processedItemsSoFar, validItems);

                // Load
                Float dataRegisters[ITEMS_PER_THREAD];
                if(validItems == DATA_PER_BLOCK) [[likely]]
                    BlockLoad().Load(dSubBlockInOut.data(), dataRegisters);
                else
                    BlockLoad().Load(dSubBlockInOut.data(), dataRegisters,
                                     validItems, Float(0));

                // Normalization
                UNROLL_LOOP
                for(uint32_t i = 0; i < ITEMS_PER_THREAD; i++)
                {
                    dataRegisters[i] *= sTotalRecip;
                }
                // Store
                if(validItems == DATA_PER_BLOCK) [[likely]]
                    BlockStore().Store(dSubBlockInOut.data(), dataRegisters);
                else
                    BlockStore().Store(dSubBlockInOut.data(), dataRegisters, validItems);

                processedItemsSoFar += validItems;
            }

            MRAY_DEVICE_BLOCK_SYNC();
        };

        // Block-stride loop (one block for each row)
        uint32_t yCount = static_cast<uint32_t>(dYCDFs.size());
        for(uint32_t block = kp.blockId; block < yCount; block += kp.gridSize)
        {
            uint32_t xCount = static_cast<uint32_t>(dXCDFs.size() / yCount);
            auto dRowCDF = dXCDFs.subspan(block * xCount, xCount);
            NormalizeRow(dRowCDF);
        }
        // Let first block to divide the thing as well
        if(kp.blockId == 0) NormalizeRow(dYCDFs);
    }

#else
    #error DistributionPwC2D kernels do not have generic implementation!
#endif


DistributionGroupPwC2D::DistributionGroupPwC2D(const GPUSystem& s)
    : system(s)
    , memory(s.AllGPUs(), 32_MiB, 64_MiB)
{}

uint32_t DistributionGroupPwC2D::Reserve(Vector2ui size)
{
    sizes.push_back(size);
    return static_cast<uint32_t>(sizes.size() - 1);
}

void DistributionGroupPwC2D::Commit()
{
    using SizeList = std::array<size_t, 4>;
    // Commit the reservations
    std::vector<SizeList> alignedSizes(sizes.size());
    std::transform(sizes.cbegin(), sizes.cend(),
                   alignedSizes.begin(),
                   [](const Vector2ui& vec) -> SizeList
    {
        return SizeList
        {
            // X CDF Data
            vec[0] * vec[1],
            // Y CDF Data
            vec[1],
            // X Dist1D Align Size
            vec[1],
            // Dist2D Itself
            1
        };
    });

    SizeList totalSizes = std::reduce(alignedSizes.cbegin(),
                                      alignedSizes.cend(),
                                      SizeList{0,0,0,0},
                                      [](const SizeList& a, const SizeList& b)
    {
        SizeList result = {};
        result[0] = a[0] + b[0];
        result[1] = a[1] + b[1];
        result[2] = a[2] + b[2];
        result[3] = a[3] + b[3];

        return std::move(result);
    });

    Span<Float> dCDFsX;
    Span<Float> dCDFsY;
    Span<DistributionPwC<1>> dDistsX;
    Span<DistributionPwC<1>> dDistsY;
    MemAlloc::AllocateMultiData(std::tie(dCDFsX, dCDFsY, dDistsX,
                                         dDistsY, dDistributions),
                                memory,
                                {totalSizes[0], totalSizes[1],
                                 totalSizes[2], totalSizes[3],
                                totalSizes[3],
                                });

    // Calculate "Pointers"
    distData.reserve(sizes.size());
    SizeList offsets = {0, 0, 0, 0};
    for(size_t i = 0; i < alignedSizes.size(); i++)
    {
        auto d = DistData
        {
            .dCDFsX = dCDFsX.subspan(offsets[0], alignedSizes[i][0]),
            .dCDFsY = dCDFsY.subspan(offsets[1], alignedSizes[i][1]),
            .dDistsX = dDistsX.subspan(offsets[2], alignedSizes[i][2]),
            .dDistY = Span<Distribution1D, 1>(dDistsY.subspan(offsets[3], 1))
        };
        offsets[0] += alignedSizes[i][0];
        offsets[1] += alignedSizes[i][1];
        offsets[2] += alignedSizes[i][2];
        offsets[3] += alignedSizes[i][3];
        distData.push_back(d);
    };

    if constexpr(MRAY_IS_DEBUG)
    {
        for(size_t i = 0; i < offsets.size(); i++)
            assert(offsets[i] == totalSizes[i]);
    }
}

void DistributionGroupPwC2D::Construct(uint32_t index,
                                       const Span<const Float>& function)
{
    using namespace DeviceAlgorithms;
    using namespace std::literals;
    assert(index < distData.size());

    // TODO: select a device?
    const GPUQueue& queue = system.BestDevice().GetQueue(0);
    const DistData& d = distData[index];

    // Directly scan to cdf array
    InclusiveMultiScan(d.dCDFsX,
                       function,
                       sizes[index][0],
                       Float{0},
                       queue,
                       []MRAY_HYBRID(Float a, Float b) {return a + b; });

    // Copy to Y and normalize
    uint32_t yCount = static_cast<uint32_t>(sizes[index][1]);
    queue.IssueExactKernel<KCCopyScanY>
    (
        "Dist2D-Copy&ScanY"sv,
        KernelExactIssueParams{.gridSize = 1, .blockSize = TPB},
        //
        d.dCDFsY,
        ToConstSpan(d.dCDFsX)
    );

    uint32_t xCount = static_cast<uint32_t>(sizes[index][0]);
    queue.IssueSaturatingKernel<KCNormalizeXY>
    (
        "Dist2D-NormalizeXY"sv,
        KernelIssueParams{.workCount = xCount * TPB},
        //
        d.dCDFsX,
        d.dCDFsY
    );

    queue.IssueSaturatingLambda
    (
        "Dist2D-ConstructDist"sv,
        KernelIssueParams{.workCount = yCount},
        [d, xCount, yCount, dDist = dDistributions.subspan(index, 1)] MRAY_GPU(KernelCallParams kp)
        {
            for(uint32_t i = kp.GlobalId(); i < yCount;
                i += kp.TotalSize())
            {
                d.dDistsX[i] = Distribution1D(ToConstSpan(d.dCDFsX.subspan(i * xCount, xCount)));
            }
            MRAY_DEVICE_BLOCK_SYNC();

            if(kp.GlobalId() == 0)
            {
                d.dDistY[0] = Distribution1D(d.dCDFsY);
                dDist[0] = Distribution(ToConstSpan(d.dDistsX), d.dDistY[0]);
            }
        }
    );
}

Span<const DistributionPwC<2>> DistributionGroupPwC2D::DeviceDistributions() const
{
    return ToConstSpan(dDistributions);
}

size_t DistributionGroupPwC2D::GPUMemoryUsage() const
{
    return memory.Size();
}

typename DistributionGroupPwC2D::DistDataConst DistributionGroupPwC2D::DistMemory(uint32_t index) const
{
    const DistData& d = distData[index];
    return DistDataConst
    {
        .dCDFsX = ToConstSpan(d.dCDFsX),
        .dCDFsY = ToConstSpan(d.dCDFsY),
        .dDistsX = ToConstSpan(d.dDistsX),
        .dDistY = ToConstSpan(d.dDistY),
    };
}

}