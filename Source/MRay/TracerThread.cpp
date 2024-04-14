#include "TracerThread.h"



void TracerThread::LoopWork()
{
    uint32_t rendererIndex = std::numeric_limits<uint32_t>::max();
    uint32_t cameraIndex = std::numeric_limits<uint32_t>::max();
    Float time = std::numeric_limits<Float>::max();
    CameraTransform newTransform = {};
    std::string newScenePath = "";

    // On every "frame", we will do the latest common commands
    VisorCommand command;
    while(transferQueue.TryDequeue(command))
    {
        // Technically this loop may not terminate,
        // if stuff comes too fast. But we just setting some data so
        // it should not be possible

        // Process command....
    }

    if(!newScenePath.empty())
    {
        tracer->ClearAll();
        tracer->
    }


    // Set reset renderer etc.


    Optional<TracerImgOutput> output = tracer->DoRenderWork();
    //if(output) transferQueue.Enqueue(...)

    }

void TracerThread::InitialWork()
{}

void TracerThread::FinalWork()
{}

TracerThread::TracerThread(TransferQueue& queue,
                           std::string& sharedLibraryPath,
                           const std::string& constructorMangledName,
                           const std::string& destructorMangledName)
    : dllFile(sharedLibraryPath)
    , tracer{nullptr, nullptr}
    , transferQueue(queue.GetTracerView())
{
    SharedLibArgs args =
    {
        constructorMangledName,
        destructorMangledName
    };
    MRayError err = dllFile.GenerateObjectWithArgs<TracerI>(tracer, args);
    if(err) throw err;
}