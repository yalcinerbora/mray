#pragma once

#include "Core/TracerI.h"
#include "Core/RealtimeThread.h"
#include "Core/SharedLibrary.h"

#include "TransferQueue.h"

class TracerThread final : public RealtimeThread
{
    private:
    std::unique_ptr<SharedLibrary>  dllFile;
    SharedLibPtr<TracerI>           tracer;
    TransferQueue::TracerView       transferQueue;
    //
    void        LoopWork() override;
    void        InitialWork() override;
    void        FinalWork() override;

    public:
    // Constructors & Destructor
                TracerThread(TransferQueue& queue);
                ~TracerThread() = default;

    MRayError   MTInitialize(const std::string& tracerConfig);
    bool        InternallyTerminated() const override;

};
