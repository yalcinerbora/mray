#include "TracerThread.h"
#include <BS/BS_thread_pool.hpp>

void TracerThread::LoopWork()
{
    uint32_t rendererIndex = std::numeric_limits<uint32_t>::max();
    uint32_t cameraIndex = std::numeric_limits<uint32_t>::max();
    Float time = std::numeric_limits<Float>::max();
    CameraTransform newTransform = {};
    std::string newScenePath = "";

    // On every "frame", we will do the latest common commands
    VisorAction command;
    while(transferQueue.TryDequeue(command))
    {
        // Technically this loop may not terminate,
        // if stuff comes too fast. But we just setting some data so
        // it should not be possible

        // Process command....
        //

    }

    if(!newScenePath.empty())
    {
        tracer->ClearAll();
    }


    // Set reset renderer etc.


    //Optional<TracerImgOutput> output = tracer->DoRenderWork();
    //if(output) transferQueue.Enqueue(...)

    }

void TracerThread::InitialWork()
{}

void TracerThread::FinalWork()
{}

TracerThread::TracerThread(TransferQueue& queue, BS::thread_pool& tp)
    : dllFile{nullptr}
    , tracer{nullptr, nullptr}
    , transferQueue(queue.GetTracerView())
    , threadPool(tp)
{}

MRayError TracerThread::MTInitialize(const std::string& tracerConfig)
{
    std::string sharedLibraryPath = "";
    std::string constructorMangledName = "";
    std::string destructorMangledName = "";

    SharedLibArgs args =
    {
        constructorMangledName,
        destructorMangledName
    };
    MRayError err = dllFile->GenerateObjectWithArgs<TracerConstructorArgs, TracerI>(tracer, args,
                                                                                    threadPool);
    if(err) return err;

    return MRayError::OK;
}

bool TracerThread::InternallyTerminated() const
{
    // TODO: Add crash condition
    return false;
}