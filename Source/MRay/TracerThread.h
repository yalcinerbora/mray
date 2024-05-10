#pragma once

#include "Core/TracerI.h"
#include "Core/RealtimeThread.h"
#include "Core/SharedLibrary.h"

#include "Common/TransferQueue.h"

namespace BS { class thread_pool; }

class TracerThread final : public RealtimeThread
{
    private:
    std::unique_ptr<SharedLibrary>  dllFile;
    SharedLibPtr<TracerI>           tracer;
    TransferQueue::TracerView       transferQueue;
    BS::thread_pool&                threadPool;

    //
    void        LoopWork() override;
    void        InitialWork() override;
    void        FinalWork() override;

    public:
    // Constructors & Destructor
                TracerThread(TransferQueue& queue,
                             BS::thread_pool&);
                ~TracerThread() = default;

    MRayError   MTInitialize(const std::string& tracerConfig);
    bool        InternallyTerminated() const override;
};
