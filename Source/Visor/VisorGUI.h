#pragma once

#include "Core/MathForward.h"
#include "Core/Vector.h"
#include "Core/Types.h"

struct ImFont;

struct TracerAnalyticData
{
    // Performance
    double          throughput;
    std::string     throughputSuffix;
    //
    double          workPerPixel;
    std::string     workPerPixelSuffix;
    // Timings
    float           iterationTimeMS;

    // Memory Related
    double          totalGPUMemoryMiB;
    double          usedGPUMemoryMiB;

    // Image related
    const Vector2i  renderResolution;
};

struct SceneAnalyticData
{
    enum SceneGroupTypes
    {
        MATERIAL,
        PRIMITIVE,
        LIGHT,
        CAMERA,
        ACCELERATOR,
        TRANSFORM,
        MEDIUM,

        END
    };

    // Generic
    std::string                 sceneName;
    // Timings
    double                      sceneLoadTime;      // secs
    double                      sceneUpdateTime;    // secs
    // Group Counts
    std::array<uint32_t, END>   groupCounts;
    // Key Maximums
    Vector2i                    accKeyMax;
    Vector2i                    workKeyMax;
};

enum class RunState
{
    RUNNING,
    STOPPED,
    PAUSED,

    END
};

using namespace std::string_view_literals;

class MainStatusBar
{
    private:

    static constexpr auto RENDERING_NAME = "Rendering"sv;
    static constexpr auto PAUSED_NAME    = "PAUSED"sv;
    static constexpr auto STOPPED_NAME   = "STOPPED"sv;

    bool    paused;
    bool    running;
    bool    stopped;

    protected:
    public:
    // Constructors & Destructor
                MainStatusBar();
                ~MainStatusBar() = default;

    Optional<RunState>    Render(const TracerAnalyticData&,
                                 const SceneAnalyticData&);
};


class VisorGUI
{
    private:
    MainStatusBar       statusBar;
    bool topBarOn       = true;
    bool bottomBarOn    = true;

    public:
    void Render(ImFont* windowScaledFont);
};