#include <atomic>

// Vulkan - CUDA interop well define only for intra GPU operations.
// However our main use-case is to synchronize between inter
// GPU synchronizations (iGPU vs MRayGPUDevice).
// So we create a simple vulkan style  timeline semaphore.
// C++20 has "wait" on atomic variables unfortunately it is not what
// we need. It waits "atomic != var" we need wait untill "atomic == var".
// We need to handroll our own wait via mutex/condition_var/uint64_t.
//
// Additionally release operation handled by another thread,
// for example, on CUDA side driver (or w/e the executor of cudaLaunchHostFunc
// is) will release when the write operation finishes. Because of that
// we need to "notify_all" instead of "notify_one" due to thread runaway issue.
// (Tracer thread started to wait on this "semaphore" without preemtion)
//
// Only drawback is that we need to wait/signal on CPU side instead of GPU side.
// This should not be an issue since we already send the data that we
// need to synchronize on Host(CPU) memory so that portion should be the
// bottleneck instead of this.
//
// TODO: With some simple tests CUDA portions fluctuates a lot (0.08ms to 2.5ms)
// with empty stream (Acquire on Host, launch the host function)
// Visor portion (similarly acquire on host, launch a thread that release)
// has somewhat consistent timing (between 0.2ms 0.5ms).
// Hopefully this is good enough when actual load is put on these systems.
// Check back later! (We may try to utilize CCCL, or busy wait with yield)
class TimelineSemaphore
{
    private:
    uint64_t                value;
    std::mutex              mutex;
    std::condition_variable cVar;
    public:
    // Constructors & Destructor
         TimelineSemaphore(uint64_t initialValue);
    //
    void Acquire(uint64_t);
    void Release();
    void Reset();
};

inline
TimelineSemaphore::TimelineSemaphore(uint64_t init)
    : value(init)
{}

inline
void TimelineSemaphore::Acquire(uint64_t waitVal)
{
    // First wait is "free" we assume it is signalled
    if(waitVal == 0) return;

    std::unique_lock lock(mutex);
    cVar.wait(lock, [this, waitVal]()
    {
        // Wait untill value is exactly the wait value
        return value == waitVal;
    });
}

inline
void TimelineSemaphore::Release()
{
    std::unique_lock lock(mutex);
    value++;
    cVar.notify_all();
}

inline
void TimelineSemaphore::Reset()
{
    std::unique_lock lock(mutex);
    value = 0;
    cVar.notify_all();
}