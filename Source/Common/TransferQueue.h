#pragma once

#include "Core/MPMCQueue.h"
#include "Core/TracerI.h"
#include "Core/TimelineSemaphore.h"
#include "Core/Variant.h"

#include "AnalyticStructs.h"
#include "RenderImageStructs.h"

struct SemaphoreInfo
{
    TimelineSemaphore*  semaphore           = nullptr;
    uint32_t            importMemAlignment  = 0;
};

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
    RenderImageSaveInfo,    // hdr save
    RenderImageSaveInfo,    // sdr save
    uint64_t                // memory usage
>
{
    using Base = std::variant<CameraTransform, SceneAnalyticData,
                         TracerAnalyticData, RendererAnalyticData,
                         RendererOptionPack, RenderBufferInfo,
                         bool, RenderImageSection, RenderImageSaveInfo,
                         RenderImageSaveInfo, uint64_t>;
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
        MEMORY_USAGE = 10           // Current memory usage
    };
    using Base::Base;
};

struct VisorAction : public std::variant
<
    CameraTransform,        // transform
    uint32_t,               // camera index
    std::string,            // renderer index
    uint32_t,               // renderer logic0 index
    uint32_t,               // renderer logic1 index
    std::string,            // scene name
    float,                  // scene time
    bool,                   // start/stop render
    bool,                   // pause render
    SemaphoreInfo,          // Synchronization semaphore
    bool,                   // Demand HDR save
    bool,                   // Demand SDR save
    std::string             // Initial Render Config
>
{
    using Base = std::variant<CameraTransform, uint32_t,
                         std::string, uint32_t, uint32_t, std::string,
                         float, bool, bool,
                         SemaphoreInfo, bool, bool,
                         std::string>;
    enum Type
    {
        CHANGE_CAM_TRANSFORM = 0,   // Give new transform to the tracer
                                    // for the current cam
        CHANGE_CAMERA = 1,          // Change to a camera via an id.
                                    // Camera list is in "SceneAnalytics" structure
                                    // Tracer will respond via a camera initial transform.
        CHANGE_RENDERER = 2,        // Change the renderer via a name. Tracer will respond
                                    // with initial parametrization of the renderer.
                                    // renderer list is in "TracerAnalytics" structure.
        CHANGE_RENDER_LOGIC0 = 3,   // Change the renderer logic0 via an index.
                                    // These logic parameters may be useful for debugging,
                                    // etc.
        CHANGE_RENDER_LOGIC1 = 4,   // Change the renderer logic1 via an index.
        LOAD_SCENE = 5,             // Load a scene, tracer will respond with a
                                    // "SceneAnalytics" struct
        CHANGE_TIME = 6,            // Change the time of the scene. Min max values are in
                                    // "SceneAnalytics" strcut
        START_STOP_RENDER = 7,      // Start stop the rendering.
        PAUSE_RENDER = 8,           // Pause the rendering
        SEND_SYNC_SEMAPHORE = 9,    // Send synchronization semaphore
        DEMAND_HDR_SAVE = 10,       // Request a save event. This goes through tracer
        DEMAND_SDR_SAVE = 11,       // because tracer knows better when to exactly save.
                                    // Renderer will trigger a save when it is either on an ~spp
                                    // boundary (or closer)
        KICKSTART_RENDER = 12       // Initial render kickstart, send renderer config
                                    // Tracer initializes the renderer via this json file
                                    // It does not start rendering though
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
        void                Dequeue(VisorAction&);
        bool                TryDequeue(VisorAction&);
        template<class D>
        TimedDequeueResult  TryDequeue(VisorAction&, D duration);
        void                Enqueue(TracerResponse&&);
        bool                TryEnqueue(TracerResponse&&);
        //
        void    Terminate();
        bool    IsTerminated() const;
    };

    class VisorView
    {
        private:
        TransferQueue&      tq;
        public:
                            VisorView(TransferQueue&);
        void                Dequeue(TracerResponse&);
        bool                TryDequeue(TracerResponse&);
        template<class D>
        TimedDequeueResult  TryDequeue(TracerResponse&, D duration);
        void                Enqueue(VisorAction&&);
        bool                TryEnqueue(VisorAction&&);
        //
        void    Terminate();
        bool    IsTerminated() const;
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
    , tracerView(*this, std::move(Trigger))
    , visorView(*this)
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

template<class D>
inline TimedDequeueResult
TransferQueue::TracerView::TryDequeue(VisorAction& vc, D duration)
{
    return tq.commands.TryDequeue(vc, duration);
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

inline void TransferQueue::TracerView::Terminate()
{
    // Terminate the queue
    tq.Terminate();
    // Visor may be in interactive mode and wait on keyevents etc..
    // trigger a refresh on window.
    VisorTrigger();
}
inline bool TransferQueue::TracerView::IsTerminated() const
{
    return tq.responses.IsTerminated() || tq.commands.IsTerminated();
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

template<class D>
inline TimedDequeueResult
TransferQueue::VisorView::TryDequeue(TracerResponse& vc, D duration)
{
    return tq.responses.TryDequeue(vc, duration);
}

inline void TransferQueue::VisorView::Enqueue(VisorAction&& vc)
{
    tq.commands.Enqueue(std::move(vc));
}

inline bool TransferQueue::VisorView::TryEnqueue(VisorAction&& vc)
{
    return tq.commands.TryEnqueue(std::move(vc));
}

inline void TransferQueue::VisorView::Terminate()
{
    tq.Terminate();
}

inline bool TransferQueue::VisorView::IsTerminated() const
{
    return tq.responses.IsTerminated() || tq.commands.IsTerminated();
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