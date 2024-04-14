#pragma once

#include "Core/TracerI.h"
#include "Core/RealtimeThread.h"
#include "Core/SharedLibrary.h"

#include "TransferQueue.h"

class TracerThread : public RealtimeThread
{
    private:
    SharedLibrary               dllFile;
    SharedLibPtr<TracerI>       tracer;
    TransferQueue::TracerView   transferQueue;
    //
    void                        LoopWork() override;
    void                        InitialWork() override;
    void                        FinalWork() override;

    public:
    // Constructors & Destructor
                TracerThread(TransferQueue& queue,
                             std::string& sharedLibraryPath,
                             const std::string& constructorMangledName,
                             const std::string& destructorMangledName);
    virtual     ~TracerThread() = default;

};
