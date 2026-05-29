#pragma once

#include "Core/DataStructures.h"

#include <string>

struct ImFont;

class FontAtlas
{
    static constexpr size_t MAX_FONT_COUNT = 16;
    // TODO: float key is scary.
    using FontList = StaticVector<Pair<float, ImFont*>, MAX_FONT_COUNT>;

    private:
    FontList            scaledFonts;
    std::string         executablePath = "";

    // Constructors & Destructor
                        FontAtlas() = default;
                        FontAtlas(std::string_view execPath);
    public:
    static FontAtlas&   Instance(std::string_view execPath = "");

    ImFont*             GetScaledFont(float scaling);
    void                AddScaledFont(float scaling);
    void                RemoveScaledFont(float scaling);
    void                ClearFonts();
};