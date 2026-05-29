#include "FontAtlas.h"
#include <Imgui/imgui.h>

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include "Core/Filesystem.h"
#include "../Resources/Fonts/IcoMoonFontTable.h"

FontAtlas::FontAtlas(std::string_view execPath)
    : executablePath(execPath)
{}

FontAtlas& FontAtlas::Instance(std::string_view execPath)
{
    static FontAtlas instance;
    if(!execPath.empty())
    {
        assert(instance.executablePath.empty());
        ImGui::GetIO().Fonts->Clear();
        instance = FontAtlas(execPath);
    }
    return instance;
}

ImFont* FontAtlas::GetScaledFont(float scaling)
{
    auto loc = std::find_if(scaledFonts.cbegin(), scaledFonts.cend(),
                            [scaling](const auto& pair)
    {
        return (scaling == pair.first);
    });
    assert(loc != scaledFonts.cend());
    return loc->second;
}

void FontAtlas::AddScaledFont(float scaling)
{
    using namespace std::string_view_literals;
    static constexpr auto VERA_MONO_FONT_PATH = "Fonts/VeraMono.ttf"sv;
    static constexpr auto ICON_FONT_PATH = std::string_view("Fonts/" FONT_ICON_FILE_NAME_ICOMN);

    // %5 bigger fonts initially looked better (subjective)
    static constexpr float INITIAL_SCALE = 1.00f;
    static constexpr float PIXEL_SIZE = 14;

    // Only add font with this scale if not available
    auto loc = std::find_if(scaledFonts.cbegin(), scaledFonts.cend(),
                            [scaling](const auto& pair)
    {
        return scaling == pair.first;
    });
    if(loc != scaledFonts.cend()) return;

    float scaledPixelSize = std::roundf(PIXEL_SIZE * INITIAL_SCALE *
                                        scaling);
    ImGuiIO& io = ImGui::GetIO();
    ImFontConfig config;
    config.RasterizerDensity = 2.0f;
    config.SizePixels = scaledPixelSize;
    config.PixelSnapH = false;
    config.OversampleH = config.OversampleV = 2;
    config.MergeMode = false;
    std::string monoTTFPath = Filesystem::RelativePathToAbsolute(VERA_MONO_FONT_PATH,
                                                                 executablePath);
    ImFont* font = io.Fonts->AddFontFromFileTTF(monoTTFPath.c_str(),
                                                config.SizePixels,
                                                &config);
    // Icomoon
    config.MergeMode = true;
    config.PixelSnapH = true;
    config.GlyphMinAdvanceX = scaledPixelSize;
    config.GlyphMaxAdvanceX = scaledPixelSize;
    config.OversampleH = config.OversampleV = 1;
    config.GlyphOffset = ImVec2(0, 4);
    static const ImWchar icon_ranges[] = {ICON_MIN_ICOMN, ICON_MAX_ICOMN, 0};
    std::string ofiTTFPath = Filesystem::RelativePathToAbsolute(ICON_FONT_PATH,
                                                                executablePath);
    [[maybe_unused]]
    ImFont* iconFont = io.Fonts->AddFontFromFileTTF(ofiTTFPath.c_str(),
                                                    scaledPixelSize,
                                                    &config, icon_ranges);
    // We are merging this must be true
    assert(font == iconFont);
    scaledFonts.emplace_back(scaling, font);
}

void FontAtlas::RemoveScaledFont(float)
{
    // TODO: There is a "memory leak" (technically not)
    // here, if user repeatedly removes adds monitors during a session
    // Imgui's atlas will become large
    // Maybe add full reclear time to time
    // This should be extremely rare though...
    //
    // Some api like this probably needed (this is probably as
    // time consuming as creating entire atlas so not provided)
    // ImGui::GetIO().Fonts->ClearFonts(loc->second);

    // Keeping this empty for now...
}

void FontAtlas::ClearFonts()
{
    // TODO: Same as above
    scaledFonts.clear();
}