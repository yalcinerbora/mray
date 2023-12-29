
#include <cuda.h>
#include <cuda_runtime.h>
#include "Core/MathFunctions.h"
#include "DefinitionsCUDA.h"

static constexpr uint32_t WarpSize()
{
    return 32u;
}

// A Good guess for TPB
static constexpr uint32_t StaticThreadPerBlock1D()
{
    return 512u;
}

// TODO: This should not be compile time static
static constexpr uint32_t TotalQueuePerDevice()
{
    return 4;
}

// Cuda Kernel Optimization Hints
// Since we call all of the kernels in a static manner
// (in case of Block Size) hint the compiler
// using __launch_bounds__ expression
#define MRAY_DEVICE_LAUNCH_BOUNDS(X) __launch_bounds__(X)
#define MRAY_DEVICE_LAUNCH_BOUNDS_1D \
        MRAY_DEVICE_LAUNCH_BOUNDS __launch_bounds__(StaticThreadPerBlock1D())

#define MRAY_GRID_CONSTANT __grid_constant__

namespace CudaKernelCalls
{
    //template <class K, class... Args>
    //MRAY_KERNEL static void KernelCallCUDA(K&&, Args&&...);

    //template <uint32_t TPB, class K, class... Args>
    //MRAY_KERNEL static void KernelCallCUDA(K&&, Args&&...);

    template <class K, class... Args>
    MRAY_KERNEL static
    void KernelCallCUDA(K&& Kernel, Args&&... args)
    {
        // Grid-stride loop
        KernelCallParameters1D launchParams
        {
            .gridSize = gridDim.x,
            .blockSize = blockDim.x,
            .blockId = blockIdx.x,
            .threadId = threadIdx.x
        };
        std::forward<K>(Kernel)(launchParams, std::forward<Args>(args)...);
    }

    template <uint32_t TPB, class K, class... Args>
    //__launch_bounds__(TPB)
    MRAY_KERNEL static
    void KernelCallCUDA(K&& Kernel, Args&&... args)
    {
        // Grid-stride loop
        KernelCallParameters1D launchParams
        {
            .gridSize = gridDim.x,
            .blockSize = blockDim.x,
            .blockId = blockIdx.x,
            .threadId = threadIdx.x
        };
        std::forward<K>(Kernel)(launchParams, std::forward<Args>(args)...);
    }
}

namespace mray::cuda
{

class GPUDeviceCUDA;

class GPUQueueCUDA
{
    private:
    cudaStream_t        stream;
    uint32_t            multiprocessorCount;

    MRAY_HYBRID
    uint32_t            DetermineGridStrideBlock(const void* kernelPtr,
                                                 uint32_t sharedMemSize,
                                                 uint32_t threadCount,
                                                 uint32_t workCount) const;

    public:
    // Constructors & Destructor
    MRAY_HYBRID                 GPUQueueCUDA(uint32_t multiprocessorCount,
                                             DeviceQueueType t = DeviceQueueType::NORMAL);
                                GPUQueueCUDA(const GPUQueueCUDA&) = delete;
    MRAY_HYBRID                 GPUQueueCUDA(GPUQueueCUDA&&);
    GPUQueueCUDA&               operator=(const GPUQueueCUDA&) = delete;
    MRAY_HYBRID GPUQueueCUDA&   operator=(GPUQueueCUDA&&);
    MRAY_HYBRID                 ~GPUQueueCUDA();

    // Classic GPU Calls
    // Create just enough blocks according to work size
    template<class Function, class... Args>
    MRAY_HYBRID void    IssueKernel(uint32_t sharedMemSize,
                                    uint32_t workCount,
                                    //
                                    Function&&, Args&&...) const;
    template<class Function, class... Args>
    MRAY_HYBRID void    IssueKernel(uint32_t workCount,
                                    //
                                    Function&&, Args&&...) const;
    // Grid-Stride Kernels
    // Kernel is launched just enough blocks to
    // fully saturate the GPU.
    template<class Function, class... Args>
    MRAY_HOST void      IssueSaturatingKernel(uint32_t sharedMemSize,
                                              uint32_t workCount,
                                              //
                                              Function&&, Args&&...) const;
    template<class Function, class... Args>
    MRAY_HOST void      IssueSaturatingKernel(uint32_t workCount,
                                              //
                                              Function&&, Args&&...) const;
    // Exact Kernel Calls
    // You 1-1 specify block and grid dimensions
    // Important: These can not be annottated with launch_bounds
    template<class Function, class... Args>
    MRAY_HYBRID void    IssueExactKernel(uint32_t sharedMemSize,
                                         uint32_t gridDim, uint32_t blockDim,
                                         //
                                         Function&&, Args&&...) const;
    template<class Function, class... Args>
    MRAY_HYBRID void    IssueExactKernel(uint32_t gridDim, uint32_t blockDim,
                                         //
                                         Function&&, Args&&...) const;

    MRAY_HYBRID
    static uint32_t     RecommendedBlockCountPerSM(const void* kernelPtr,
                                                   uint32_t threadsPerBlock,
                                                   uint32_t sharedMemSize);
};

class GPUDeviceCUDA
{
    using DeviceQueues = std::vector<GPUQueueCUDA>;

    private:
    int                     deviceId;
    cudaDeviceProp          props;
    DeviceQueues            queues;

    protected:
    public:
    // Constructors & Destructor
    explicit                GPUDeviceCUDA(int deviceId);
                            GPUDeviceCUDA(const GPUDeviceCUDA&) = delete;
                            GPUDeviceCUDA(GPUDeviceCUDA&&) = default;
    GPUDeviceCUDA&          operator=(const GPUDeviceCUDA&) = delete;
    GPUDeviceCUDA&          operator=(GPUDeviceCUDA&&) = default;
                            ~GPUDeviceCUDA() = default;

    int                     DeviceId() const;
    std::string             Name() const;
    std::string             ComputeCapability() const;
    size_t                  TotalMemory() const;

    uint32_t                SMCount() const;
    uint32_t                MaxActiveBlockPerSM(uint32_t threadsPerBlock = StaticThreadPerBlock1D()) const;

    const GPUQueueCUDA&     GetQueue(uint32_t index) const;
};

class GPUSystemCUDA
{
    public:
    using GPUList = std::vector<GPUDeviceCUDA>;

    private:
    GPUList                 systemGPUs;

    protected:
    public:
    // Constructors & Destructor
                            GPUSystemCUDA();
                            GPUSystemCUDA(const GPUSystemCUDA&) = delete;
                            GPUSystemCUDA(GPUSystemCUDA&&) = delete;
    GPUSystemCUDA&          operator=(const GPUSystemCUDA&) = delete;
    GPUSystemCUDA&          operator=(GPUSystemCUDA&&) = delete;

    // Multi-Device Splittable Smart GPU Calls
    // Automatic device split and stream split on devices
    std::vector<size_t>     SplitWorkToMultipleGPU(uint32_t workCount,
                                                   uint32_t threadCount,
                                                   uint32_t sharedMemSize,
                                                   void* f) const;

    // Misc
    const GPUList&          SystemDevices() const;
    const GPUDeviceCUDA&    BestDevice() const;

    // Get Kernel Attributes & Set Dynamic Shared Memory Size
    KernelAttributes        GetKernelAttributes(const void* kernelPtr) const;
    bool                    SetKernelShMemSize(const void* kernelPtr,
                                               int sharedMemConfigSize) const;

    size_t                  TotalMemory() const;

    // Simple & Slow System Synchronization
    void                    SyncAll() const;
};

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t GPUQueueCUDA::RecommendedBlockCountPerSM(const void* kernelPtr,
                                                  uint32_t threadsPerBlock,
                                                  uint32_t sharedMemSize)
{
    int numBlocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
                                                             kernelPtr,
                                                             threadsPerBlock,
                                                             sharedMemSize));
    return static_cast<uint32_t>(numBlocks);
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t GPUQueueCUDA::DetermineGridStrideBlock(const void* kernelPtr,
                                                uint32_t sharedMemSize,
                                                uint32_t threadCount,
                                                uint32_t workCount) const
{
    // TODO: Make better SM determination
    uint32_t blockPerSM = RecommendedBlockCountPerSM(kernelPtr, threadCount, sharedMemSize);
    // Only call enough SM
    uint32_t totalRequiredBlocks = MathFunctions::AlignToMultiple(workCount, threadCount);
    uint32_t requiredSMCount = (totalRequiredBlocks + blockPerSM - 1) / blockPerSM;
    uint32_t smCount = std::min(multiprocessorCount, requiredSMCount);
    uint32_t blockCount = std::min(requiredSMCount, smCount * blockPerSM);
    return blockCount;
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCUDA::GPUQueueCUDA(uint32_t multiprocessorCount,
                           DeviceQueueType t)
    : multiprocessorCount(multiprocessorCount)
{
    // CudaStream Create
    switch(t)
    {
        case DeviceQueueType::NORMAL:
            CUDA_CHECK(cudaStreamCreate(&stream));
            break;
        case DeviceQueueType::FIRE_AND_FORGET:
            stream = cudaStreamFireAndForget;
            break;
        case DeviceQueueType::TAIL_LAUNCH:
            stream = cudaStreamTailLaunch;
            break;
        default:
            assert(false);
            break;
    }
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCUDA::GPUQueueCUDA(GPUQueueCUDA&& other)
    : stream(other.stream)
    , multiprocessorCount(other.multiprocessorCount)
{
    other.stream = (cudaStream_t)0;
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCUDA& GPUQueueCUDA::operator=(GPUQueueCUDA&& other)
{
    multiprocessorCount = other.multiprocessorCount;
    stream = other.stream;
    other.stream = (cudaStream_t)0;
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCUDA::~GPUQueueCUDA()
{
    if(stream != cudaStreamTailLaunch ||
       stream != cudaStreamFireAndForget ||
       stream != (cudaStream_t)0)
    {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

template<class Function, class... Args>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueKernel(uint32_t sharedMemSize, uint32_t workCount,
                               Function&& f, Args&&... args) const
{
    using namespace CudaKernelCalls;
    uint32_t blockCount = MathFunctions::AlignToMultiple(workCount, StaticThreadPerBlock1D());
    uint32_t blockSize = StaticThreadPerBlock1D();

    KernelCallCUDA<StaticThreadPerBlock1D()>
    <<<blockCount, blockSize, sharedMemSize, stream>>>
    (
        std::forward<Function>(f),
        std::forward<Args>(args)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
MRAY_HOST inline
void GPUQueueCUDA::IssueKernel(uint32_t workCount,
                               Function&& f, Args&&... args) const
{
    KC_X(0, workCount, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
MRAY_HOST inline
void GPUQueueCUDA::IssueSaturatingKernel(uint32_t sharedMemSize,
                                         uint32_t workCount,
                                         //
                                         Function&& f, Args&&... args) const
{
    using namespace CudaKernelCalls;

    const void* kernelPtr = reinterpret_cast<const void*>
        (&KernelCallCUDA<StaticThreadPerBlock1D(), Function, Args...>);
    uint32_t threadCount = StaticThreadPerBlock1D();
    uint32_t blockCount = DetermineGridStrideBlock(kernelPtr,
                                                   sharedMemSize,
                                                   threadCount,
                                                   workCount);


    KernelCallCUDA<StaticThreadPerBlock1D(), Function, Args...>
    <<<blockCount, threadCount, sharedMemSize, stream>>>
    (
        std::forward<Function>(f),
        std::forward<Args>(args)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
MRAY_HOST inline
void GPUQueueCUDA::IssueSaturatingKernel(uint32_t workCount,
                                         //
                                         Function&& f, Args&&... args) const
{
    SaturatingKC_X(0, workCount,
                   std::forward<Function>(f),
                   std::forward<Args>(args)...);
}

template<class Function, class... Args>
MRAY_HOST inline
void GPUQueueCUDA::IssueExactKernel(uint32_t sharedMemSize,
                                    uint32_t blockCount, uint32_t blockSize,
                                    //
                                    Function&& f, Args&&... args) const
{
    using namespace CudaKernelCalls;

    KernelCallCUDA
    <<<blockCount, blockSize, sharedMemSize, stream>>>
    (
        std::forward<Function>(f),
        std::forward<Args>(args)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
MRAY_HOST inline
void GPUQueueCUDA::IssueExactKernel(uint32_t blockCount, uint32_t blockSize,
                                    //
                                    Function&& f, Args&&... args) const
{
    ExactKC_X(0, blockCount, blockSize,
              std::forward<Function>(f),
              std::forward<Args>(args)...);
}

}


