#include <atomic>

// Vulkan - CUDA interop well define only for intra GPU operations.
// However our main use-case is to synchronize between inter
// GPU synchronizations (iGPU vs MRayGPUDevice).
// So we create a simple vulkan style  timeline semaphore.
// C++20 allows "wait" on atomic variables and thankfully this is
// the exact functionality that we need. Only drawback is that
// we need to wait/signal on CPU side instead of GPU side.
// This should not be an issue since we already send the data that we
// need to synchronize on Host(CPU) memory so that portion should be the
// bottleneck instead of this.
class TimelineSemaphore
{
    std::atomic_uint64_t  value;
    public:
    // Constructors & Destructor
         TimelineSemaphore(uint64_t initialValue);
    //
    void Acquire(uint64_t);
    void Release();
};

inline
TimelineSemaphore::TimelineSemaphore(uint64_t init)
    : value(init)
{}

inline
void TimelineSemaphore::Acquire(uint64_t waitVal)
{
    value.wait(waitVal);
}

inline
void TimelineSemaphore::Release()
{
    value.fetch_add(1);
    value.notify_one();
}