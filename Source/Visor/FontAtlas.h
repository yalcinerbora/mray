#pragma once

#include "Core/DataStructures.h"

struct GLFWmonitor;
struct ImFont;

class FontAtlas
{
    static constexpr size_t MAX_FONT_COUNT = 16;
    // TODO: float key is scary.
    using FontList = StaticVector<Pair<float, ImFont*>, MAX_FONT_COUNT>;

    private:
    FontList            monitorFonts;
    std::string         executablePath = "";

    // Constructors & Destructor
                        FontAtlas() = default;
                        FontAtlas(std::string_view execPath);
    public:
    static FontAtlas&   Instance(std::string_view execPath = "");

    ImFont*             GetMonitorFont(float scalingX);
    void                AddMonitorFont(GLFWmonitor*);
    void                RemoveMonitorFont(GLFWmonitor*);
    void                ClearFonts();
};