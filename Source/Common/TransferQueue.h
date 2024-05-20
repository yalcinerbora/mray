#pragma once

#include "Core/MPMCQueue.h"
#include "Core/RealtimeThread.h"
#include "Core/TracerI.h"

#include "TransientPool/TransientPool.h"

#include "AnalyticStructs.h"
#include "RenderImageStructs.h"

struct TracerResponse : public std::variant
<
    CameraTransform,        // initial cam transform
    SceneAnalyticData,      // scene analytics
    TracerAnalyticData,     // tracer analytics
    RendererAnalyticData,   // renderer analytics
    RendererOptionPack,     // renderer options;
    RenderBufferInfo,       // render output information
    bool,
    RenderImageSection,     // image section;
    RenderImageSaveInfo,
    RenderImageSaveInfo
>
{
    using Base = std::variant<CameraTransform, SceneAnalyticData,
                              TracerAnalyticData, RendererAnalyticData,
                              RendererOptionPack, RenderBufferInfo,
                              bool, RenderImageSection, RenderImageSaveInfo,
                              RenderImageSaveInfo>;
    enum Type
    {
        CAMERA_INIT_TRANSFORM = 0,  // Return a camera initial transform when
                                    // camera is changed
        SCENE_ANALYTICS = 1,        // scene analytics
        TRACER_ANALYTICS = 2,       // tracer analytics
        RENDERER_ANALYTICS = 3,     // renderer analytics
        RENDERER_OPTIONS = 4,       // renderer options

        RENDER_BUFFER_INFO = 5,     // Output buffer information
        CLEAR_IMAGE_SECTION = 6,    // Clear the image section
        IMAGE_SECTION = 7,          // Respond with an image section, this
                                    // may or may not be the entire responsible
                                    // portion. Image sections may be streamed
                                    // continiously.
        SAVE_AS_HDR = 8,            // Save the image to the disk
        SAVE_AS_SDR = 9,            // HDR or SDR (Tonemapped)
    };
    using Base::Base;
};

struct VisorAction : public std::variant
<
    CameraTransform,        // transform
    uint32_t,               // camera index
    uint32_t,               // renderer index
    std::string,            // scene name
    float,                  // scene time
    bool,                   // start/stop render
    bool,                   // pause render
    SystemSemaphoreHandle   // Synchronization semaphore
>
{
    using Base = std::variant<CameraTransform, uint32_t,
                              uint32_t, std::string,
                              float, bool, bool,
                              SystemSemaphoreHandle>;
    enum Type
    {
        CHANGE_CAM_TRANSFORM = 0,   // Give new transform to the tracer
                                    // for the current cam
        CHANGE_CAMERA = 1,          // Change to a camera via an id.
                                    // Camera list is in "SceneAnalytics" structure
                                    // Tracer will respond via a camera initial transform.
        CHANGE_RENDERER = 2,        // Change the renderer via an index. Tracer will respond
                                    // with initial parametrization of the renderer.
                                    // renderer list is in "TracerAnalytics" structure.
        LOAD_SCENE = 3,             // Load a scene, tracer will respond with a
                                    // "SceneAnalytics" struct
        CHANGE_TIME = 4,            // Change the time of the scene. Min max values are in
                                    // "SceneAnalytics" strcut
        START_STOP_RENDER = 5,      // Start stop the rendering.
        PAUSE_RENDER = 6,           // Pause the rendering
        SEND_SYNC_SEMAPHORE = 7     // Send synchronization semaphore
    };
    using Base::Base;
};

class TransferQueue
{
    using VisorTriggerCommand = std::function<void()>;
    public:
    class TracerView
    {
        private:
        TransferQueue&      tq;
        VisorTriggerCommand VisorTrigger;

        public:
                        TracerView(TransferQueue&,
                                   VisorTriggerCommand&&);
        void            Dequeue(VisorAction&);
        bool            TryDequeue(VisorAction&);
        void            Enqueue(TracerResponse&&);
        bool            TryEnqueue(TracerResponse&&);
    };

    class VisorView
    {
        private:
        TransferQueue&  tq;
        public:
                        VisorView(TransferQueue&);
        void            Dequeue(TracerResponse&);
        bool            TryDequeue(TracerResponse&);
        void            Enqueue(VisorAction&&);
        bool            TryEnqueue(VisorAction&&);
    };

    private:
    MPMCQueue<VisorAction>      commands;
    MPMCQueue<TracerResponse>   responses;
    TracerView                  tracerView;
    VisorView                   visorView;

    public:
    // Constructors & Destructor
                    TransferQueue(size_t commandQueueSize,
                                  size_t respondQueueSize,
                                  VisorTriggerCommand VisorTrigger);

    TracerView&     GetTracerView();
    VisorView&      GetVisorView();
    void            Terminate();
};

inline TransferQueue::TransferQueue(size_t commandQueueSize,
                                    size_t respondQueueSize,
                                    VisorTriggerCommand Trigger)
    : commands(commandQueueSize)
    , responses(respondQueueSize)
    , visorView(*this)
    , tracerView(*this, std::move(Trigger))
{}

inline TransferQueue::TracerView::TracerView(TransferQueue& t,
                                             VisorTriggerCommand&& Trigger)
    : tq(t)
    , VisorTrigger(Trigger)
{}

inline void TransferQueue::TracerView::Dequeue(VisorAction& vc)
{
    tq.commands.Dequeue(vc);
}

inline bool TransferQueue::TracerView::TryDequeue(VisorAction& vc)
{
    return tq.commands.TryDequeue(vc);
}

inline void TransferQueue::TracerView::Enqueue(TracerResponse&& tr)
{
    tq.responses.Enqueue(std::move(tr));
    VisorTrigger();
}

inline bool TransferQueue::TracerView::TryEnqueue(TracerResponse&& tr)
{
    bool result = tq.responses.TryEnqueue(std::move(tr));
    VisorTrigger();
    return result;
}

inline TransferQueue::VisorView::VisorView(TransferQueue& t)
    : tq(t)
{}

inline void TransferQueue::VisorView::Dequeue(TracerResponse& tr)
{
    tq.responses.Dequeue(tr);
}

inline bool TransferQueue::VisorView::TryDequeue(TracerResponse& tr)
{
    return tq.responses.TryDequeue(tr);
}

inline void TransferQueue::VisorView::Enqueue(VisorAction&& vc)
{
    tq.commands.Enqueue(std::move(vc));
}

inline bool TransferQueue::VisorView::TryEnqueue(VisorAction&& vc)
{
    return tq.commands.TryEnqueue(std::move(vc));
}

inline TransferQueue::TracerView& TransferQueue::GetTracerView()
{
    return tracerView;
}

inline TransferQueue::VisorView& TransferQueue::GetVisorView()
{
    return visorView;
}

inline void TransferQueue::Terminate()
{
    commands.Terminate();
    responses.Terminate();
}