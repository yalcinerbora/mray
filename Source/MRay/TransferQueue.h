#pragma once

#include "Core/MPMCQueue.h"
#include "Core/RealtimeThread.h"

class CameraTransform
{
    Vector3f position;
    Vector3f gazePoint;
    Vector3f up;
};

class TracerResponse
{
    enum Type
    {
        CAMERA_INITIAL_TRANSFORM,
        // Analytic Related
        SCENE_ANALYTICS,
        TRACER_ANALYTICS,
        RENDERER_ANALYTICS,
        RENDERER_OPTIONS,
        //
        IMAGE_COLOR_SPACE,
        IMAGE_SECTION
    };
};

class VisorCommand
{
    public:
    enum Type
    {
        CHANGE_CAM_TRANSFORM,
        CHANGE_CAMERA,
        CHANGE_RENDERER,
        LOAD_SCENE,
        CHANGE_TIME,
        START_STOP_RENDER,
        PAUSE_RENDER
    };
};

class TransferQueue
{
    public:
    class TracerView
    {
        private:
        TransferQueue&  tq;

        public:
                        TracerView(TransferQueue&);
        void            Dequeue(VisorCommand&);
        bool            TryDequeue(VisorCommand&);
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
        void            Enqueue(VisorCommand&&);
        bool            TryEnqueue(VisorCommand&&);
    };

    private:
    MPMCQueue<VisorCommand>     commands;
    MPMCQueue<TracerResponse>   responses;

    public:
    // Constructors & Destructor
    TransferQueue(size_t commandQueueSize,
                  size_t respondQueueSize);

    TracerView  GetTracerView();
    VisorView   GetVisorView();
};

inline TransferQueue::TransferQueue(size_t commandQueueSize,
                                    size_t respondQueueSize)
    : commands(commandQueueSize)
    , responses(respondQueueSize)
{}

inline TransferQueue::TracerView::TracerView(TransferQueue& t)
    : tq(t)
{}

inline void TransferQueue::TracerView::Dequeue(VisorCommand& vc)
{
    tq.commands.Dequeue(vc);
}

inline bool TransferQueue::TracerView::TryDequeue(VisorCommand& vc)
{
    return tq.commands.TryDequeue(vc);
}

inline void TransferQueue::TracerView::Enqueue(TracerResponse&& tr)
{
    tq.responses.Enqueue(std::move(tr));
}

inline bool TransferQueue::TracerView::TryEnqueue(TracerResponse&& tr)
{
    return tq.responses.TryEnqueue(std::move(tr));
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

inline void TransferQueue::VisorView::Enqueue(VisorCommand&& vc)
{
    tq.commands.Enqueue(std::move(vc));
}

inline bool TransferQueue::VisorView::TryEnqueue(VisorCommand&& vc)
{
    return tq.commands.TryEnqueue(std::move(vc));
}

TransferQueue::TracerView TransferQueue::GetTracerView()
{
    return TracerView(*this);
}

TransferQueue::VisorView TransferQueue::GetVisorView()
{
    return VisorView(*this);
}