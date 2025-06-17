#include <fstream>
#include <memory_resource>
#include <array>
#include <span>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <thread>

#include "Core/Log.h"
#include "Core/MemAlloc.h"
#include "Core/Timer.h"

#include "Templates.h"

namespace pmr = std::pmr;

static constexpr auto KERNEL_FILE_FMT = "_GEN_Kernels{}.cu"sv;
static constexpr auto TYPEGEN_HEADER_FILE = "_GEN_RequestedTypes.h"sv;
static constexpr auto TYPEGEN_RENDER_HEADER_FILE = "_GEN_RequestedRenderers.h"sv;

auto globalAllocator = pmr::monotonic_buffer_resource(2_MiB);

struct BasicLine
{
    using FilterList = pmr::vector<std::string_view>;
    std::string_view    typeName;
    std::string_view    headerFile;
    FilterList          primFilters     = FilterList(&globalAllocator);
    FilterList          transFilters    = FilterList(&globalAllocator);
    FilterList          matFilters      = FilterList(&globalAllocator);
};

struct AccelLine
{
    enum AccelType
    {
        LIN,
        BVH,
        HW
    };
    //
    AccelType           type;
    std::string_view    baseName;
    std::string_view    groupName;
    std::string_view    headerFile;
};

struct RendererLine
{
    uint32_t            workCount;
    uint32_t            lightWorkCount;
    uint32_t            camWorkCount;
    std::string_view    typeName;
    std::string_view    headerFile;
    std::string_view    workOverload;
    std::string_view    lightWorkOverload;
    std::string_view    cameraWorkOverload;
};

using PrimLine  = BasicLine;
using MatLine   = BasicLine;
using TransLine = BasicLine;
using CamLine   = BasicLine;
using MedLine   = BasicLine;
using LightLine = BasicLine;

struct LinePack
{
    pmr::vector<PrimLine> prims        = pmr::vector<PrimLine>(&globalAllocator);
    pmr::vector<MatLine> mats          = pmr::vector<MatLine>(&globalAllocator);
    pmr::vector<TransLine> trans       = pmr::vector<TransLine>(&globalAllocator);
    pmr::vector<CamLine> cams          = pmr::vector<CamLine>(&globalAllocator);
    pmr::vector<MedLine> meds          = pmr::vector<MedLine>(&globalAllocator);
    pmr::vector<LightLine> lights      = pmr::vector<LightLine>(&globalAllocator);
    pmr::vector<AccelLine> accels      = pmr::vector<AccelLine>(&globalAllocator);
    pmr::vector<RendererLine> renders  = pmr::vector<RendererLine>(&globalAllocator);
    //
    LinePack()
    {
        prims.reserve(size_t(128));
        mats.reserve(size_t(128));
        trans.reserve(size_t(128));
        cams.reserve(size_t(128));
        meds.reserve(size_t(128));
        lights.reserve(size_t(128));
        accels.reserve(size_t(128));
        renders.reserve(size_t(128));
    }
};

std::string_view StripComments(const std::string_view line)
{
    size_t loc = std::min(line.size(), line.find_first_of('#'));
    return line.substr(0, loc);
}

struct SplitStringView
{
    private:
    std::string_view    str;
    std::string_view    splitChars;
    size_t              start;
    size_t              end;

    public:
    SplitStringView(std::string_view str, std::string_view sc)
        : str(str), splitChars(sc)
        , start(str.find_first_not_of(splitChars))
        , end(str.find_first_of(splitChars, start))
    {
        end = std::min(end, str.size());
    }

    void Next()
    {
        start = str.find_first_not_of(splitChars, end);
        start = std::min(start, str.size());
        end = str.find_first_of(splitChars, start);
        end = std::min(end, str.size());
    }

    bool HasValue() const
    {
        return end != start;
    }

    std::string_view Str() const
    {
        if(end == start) return {};
        return str.substr(start, end - start);
    }
};

template<class... Args>
std::string_view FormatToGlobalBuffer(size_t sizeHint,
                                      fmt::format_string<Args...> fmtStr, Args&&... args)
{
    // Changing since pmr::string may have sso
    // while vector does not
    pmr::vector<char> fmtBuffer(&globalAllocator);
    fmtBuffer.reserve(sizeHint + sizeof...(Args) * 256);
    fmt::format_to(std::back_inserter(fmtBuffer),
                   fmtStr, std::forward<Args>(args)...);

    auto result = std::string_view(fmtBuffer.data(), fmtBuffer.size());
    return result;
}

void ParseBasicLines(pmr::vector<BasicLine>& lines,
                     std::string_view section)
{
    BasicLine result;
    int i = 0;
    for(auto ssv = SplitStringView(section, " \t"sv); ssv.HasValue(); ssv.Next(), i++)
    {
        auto line = ssv.Str();
        if(i == 0)
            continue;
        if(i == 1)
            result.typeName = line;
        else if(i == 2)
            result.headerFile = line;
        else
        {
            auto splitLoc = line.find_first_of(':');
            if(splitLoc == std::string_view::npos)
            {
                fmt::println(stderr, "Unable to parse filter \"{}\"", line);
                std::exit(1);
            }
            auto left = line.substr(0, splitLoc);
            auto right = line.substr(splitLoc + 1);
            if(left == "Primitive"sv)
                result.primFilters.push_back(right);
            else if(left == "Material"sv)
                result.matFilters.push_back(right);
            else if(left == "Transform"sv)
                result.transFilters.push_back(right);
            else
            {
                fmt::println(stderr, "Unkown concept \"{}\" on a filter", line);
                std::exit(1);
            }
        }
    }
    lines.push_back(result);
}

void ParseAccelLines(pmr::vector<AccelLine>& lines,
                       std::string_view section)
{
    AccelLine result;
    int i = 0;
    for(auto ssv = SplitStringView(section, " \t"sv); ssv.HasValue(); ssv.Next(), i++)
    {
        auto line = ssv.Str();
        if(i == 0)
            continue;
        else if(i == 1)
        {
            if(line == "LIN"sv)
                result.type = AccelLine::LIN;
            else if(line == "BVH"sv)
                result.type = AccelLine::BVH;
            else if(line == "HW"sv)
                result.type = AccelLine::HW;
            else
            {
                fmt::println(stderr, "Unkown accelerator type \"{}\"", line);
                std::exit(1);
            }

        }
        else if(i == 2)
            result.baseName = line;
        else if(i == 3)
            result.groupName = line;
        else if(i == 4)
            result.headerFile = line;
    }
    lines.push_back(result);
}

void ParseRendererLines(pmr::vector<RendererLine>& lines,
                          std::string_view section)
{
    RendererLine result;
    int i = 0;
    for(auto ssv = SplitStringView(section, " \t"sv); ssv.HasValue(); ssv.Next(), i++)
    {
        auto line = ssv.Str();
        if(i == 0)
            continue;
        else if(i == 1)
            result.typeName = line;
        else if(i == 2)
            result.headerFile = line;
        else if(i == 3)
            result.workOverload = line;
        else if(i == 4)
            result.lightWorkOverload = line;
        else if(i == 5)
            result.cameraWorkOverload = line;
        else if(i == 6)
        {
            uint32_t val = 0;
            if(std::from_chars(line.data(), line.data() + line.size(), val).ec != std::errc{})
            {
                fmt::println(stderr, "Unkown number for renderer work count ({})", line);
                std::exit(1);
            }
            result.workCount = val;
        }
        else if(i == 7)
        {
            uint32_t val = 0;
            if(std::from_chars(line.data(), line.data() + line.size(), val).ec != std::errc{})
            {
                fmt::println(stderr, "Unkown number for renderer light work count ({})", line);
                std::exit(1);
            }
            result.lightWorkCount = val;
        }
        else if(i == 8)
        {
            uint32_t val = 0;
            if(std::from_chars(line.data(), line.data() + line.size(), val).ec != std::errc{})
            {
                fmt::println(stderr, "Unkown number for renderer camera work count ({})", line);
                std::exit(1);
            }
            result.camWorkCount = val;
        }
    }
    if(i != 9)
    {
        fmt::println(stderr, "Unable to parse renderer line \"{}\" properly", section);
        std::exit(1);
    }
    lines.push_back(result);
}

void ParseTypes(LinePack& lp, const pmr::string& data)
{
    enum LineType
    {
        PRIMITIVE,
        MATERIAL,
        TRANSFORM,
        CAMERA,
        MEDIUM,
        LIGHT,
        ACCELERATOR,
        RENDERER
    };
    static constexpr std::array TYPE_TAG_NAMES =
    {
        "P"sv,
        "Mt"sv,
        "T"sv,
        "C"sv,
        "Md"sv,
        "L"sv,
        "A"sv,
        "R"sv
    };

    std::string_view section = data;
    for(auto ssv = SplitStringView(section, "\r\n"sv); ssv.HasValue(); ssv.Next())
    {
        auto line = ssv.Str();
        line = StripComments(line);
        if(line.empty()) continue;

        auto tag = line.substr(0, line.find_first_of("\t "));
        //
             if(tag == TYPE_TAG_NAMES[PRIMITIVE])   ParseBasicLines(lp.prims, line);
        else if(tag == TYPE_TAG_NAMES[MATERIAL])    ParseBasicLines(lp.mats, line);
        else if(tag == TYPE_TAG_NAMES[TRANSFORM])   ParseBasicLines(lp.trans, line);
        else if(tag == TYPE_TAG_NAMES[CAMERA])      ParseBasicLines(lp.cams, line);
        else if(tag == TYPE_TAG_NAMES[MEDIUM])      ParseBasicLines(lp.meds, line);
        else if(tag == TYPE_TAG_NAMES[LIGHT])       ParseBasicLines(lp.lights, line);
        else if(tag == TYPE_TAG_NAMES[ACCELERATOR]) ParseAccelLines(lp.accels, line);
        else if(tag == TYPE_TAG_NAMES[RENDERER])    ParseRendererLines(lp.renders, line);
    }
}

void ConditionallyWriteToFile(std::filesystem::path loc,
                              std::string_view contents)
{
    // TODO: Read old file hash and check with the current file
    // maybe?
    std::ofstream file(loc);
    file << contents;
}

bool LookFilterAndSkip(const auto& f, auto t)
{
    if((!f.empty()) && std::find(f.begin(), f.end(), t.typeName) == f.end())
        return true;
    return false;
};

void FilterAndGenIncludes(pmr::string& includes,
                          pmr::string& renderIncludes,
                          const LinePack& lp)
{
    size_t totalHeaderSize = (lp.prims.size() +
                              lp.mats.size() +
                              lp.trans.size() +
                              lp.cams.size() +
                              lp.meds.size() +
                              lp.lights.size() +
                              lp.accels.size());

    pmr::vector<std::string_view> list(&globalAllocator); list.reserve(totalHeaderSize);
    for(const auto& x : lp.prims)   list.push_back(x.headerFile);
    for(const auto& x : lp.mats)    list.push_back(x.headerFile);
    for(const auto& x : lp.trans)   list.push_back(x.headerFile);
    for(const auto& x : lp.cams)    list.push_back(x.headerFile);
    for(const auto& x : lp.meds)    list.push_back(x.headerFile);
    for(const auto& x : lp.lights)  list.push_back(x.headerFile);
    for(const auto& x : lp.accels)
    {
        if(x.type == AccelLine::HW) continue;

        list.push_back(x.headerFile);
    }
    std::sort(list.begin(), list.end());
    auto loc = std::unique(list.begin(), list.end());

    auto validRange = std::span(list.begin(), loc);
    includes.reserve(4096);
    for(const auto& h : validRange)
    {
        if(!includes.empty())
            includes += '\n';
        includes += "#include "sv;
        includes += h;
    }

    list.clear();
    for(const auto& x : lp.renders) list.push_back(x.headerFile);
    std::sort(list.begin(), list.end());
    loc = std::unique(list.begin(), list.end());
    validRange = std::span(list.begin(), loc);

    renderIncludes.reserve(4096);
    for(const auto& h : validRange)
    {
        if(!renderIncludes.empty())
            renderIncludes += '\n';
        renderIncludes += "#include "sv;
        renderIncludes += h;
    }
}

void GenMetaLightTemplates(pmr::string& result, const LinePack& lp)
{
    static constexpr auto META_LIGHT_TEMPLATE_FMT = "PackedTypes<{}, {}>"sv;
    result.reserve(4096);
    //
    for(const auto& l : lp.lights)
    for(const auto& t : lp.trans)
    {
        if(LookFilterAndSkip(l.transFilters, t))
            continue;

        if(!result.empty()) result += ",\n";
        result += "    "sv;
        result += FormatToGlobalBuffer(META_LIGHT_TEMPLATE_FMT.size(),
                                       META_LIGHT_TEMPLATE_FMT,
                                       l.typeName, t.typeName);;
    }
}

void GenAcceleratorTemplates(pmr::string& groups,
                             pmr::string& works, const LinePack& lp)
{
    static constexpr auto ACCEL_TEMPLATE_FMT = "Group<{}>"sv;
    static constexpr auto ACCEL_WORK_TEMPLATE_FMT = "AcceleratorWork<Group<{}>, {}>"sv;
    groups.reserve(4096);
    works.reserve(4096);
    //
    for(const auto& p : lp.prims)
    {
        if(p.typeName == "PrimGroupEmpty"sv) continue;

        if(!groups.empty()) groups += ",\n        "sv;
        else groups += "    "sv;
        groups += FormatToGlobalBuffer(ACCEL_TEMPLATE_FMT.size(),
                                       ACCEL_TEMPLATE_FMT, p.typeName);

        for(const auto& t : lp.trans)
        {
            if(LookFilterAndSkip(p.transFilters, t))
                continue;
            if(LookFilterAndSkip(t.primFilters, p))
                continue;

            if(!works.empty()) works += ",\n        "sv;
            else works += "    "sv;
            works += FormatToGlobalBuffer(ACCEL_WORK_TEMPLATE_FMT.size(),
                                          ACCEL_WORK_TEMPLATE_FMT,
                                          p.typeName, t.typeName);
        }
    }
}

void FindAccelNames(std::array<std::string_view, 2>& linAccelNamePair,
                    std::array<std::string_view, 2>& bvhAccelNamePair,
                    std::array<std::string_view, 2>& hwAccelNamePair,
                    pmr::string& guardedInclude,
                    const LinePack& lp)
{
    auto Find = [&](std::array<std::string_view, 2>& out,
                    AccelLine::AccelType t)
    {
        auto loc = std::find_if(lp.accels.begin(),
                                lp.accels.end(),
                                [&](const AccelLine& a)
        {
            return a.type == t;
        });

        if(loc == lp.accels.end())
        {
            fmt::println(stderr, "No linear-marked accelerator found in input file");
            std::exit(1);
        };
        out[0] = loc->baseName;
        out[1] = loc->groupName;

        if(t == AccelLine::HW)
            guardedInclude = loc->headerFile;
    };

    Find(linAccelNamePair, AccelLine::LIN);
    Find(bvhAccelNamePair, AccelLine::BVH);
    Find(hwAccelNamePair, AccelLine::HW);
}

void GenRenderWorkTemplates(pmr::string& works,
                            pmr::string& lightWorks,
                            pmr::string& camWorks,
                            const LinePack& lp)
{
    static constexpr auto WORK_TEMPLATE_FMT = "RenderWorkT<Renderer, {}, {}, {}>"sv;
    static constexpr auto LIGHT_WORK_TEMPLATE_FMT = "RenderLightWorkT<Renderer, {}, {}>"sv;
    static constexpr auto CAM_WORK_TEMPLATE_FMT = "RenderCameraWorkT<Renderer, {}, {}>"sv;
    works.reserve(4096);
    lightWorks.reserve(4096);
    camWorks.reserve(4096);
    // SCATTER WORKS
    for(const auto& p : lp.prims)
    {
        if(p.typeName == "PrimGroupEmpty"sv) continue;
        for(const auto& m : lp.mats)
        {
            if(LookFilterAndSkip(p.matFilters, m)) continue;
            if(LookFilterAndSkip(m.primFilters, p)) continue;

            for(const auto& t : lp.trans)
            {
                if(LookFilterAndSkip(p.transFilters, t)) continue;
                if(LookFilterAndSkip(m.transFilters, t)) continue;
                if(LookFilterAndSkip(t.primFilters, p)) continue;
                if(LookFilterAndSkip(t.matFilters, m)) continue;

                if(!works.empty()) works += ",\n        "sv;
                else works += "    "sv;
                works += FormatToGlobalBuffer(WORK_TEMPLATE_FMT.size(),
                                              WORK_TEMPLATE_FMT, p.typeName,
                                              m.typeName, t.typeName);;
            }
        }
    }
    // LIGHT WORKS
    for(const auto& l : lp.lights)
    for(const auto& t : lp.trans)
    {
        if(LookFilterAndSkip(l.transFilters, t)) continue;

        if(!lightWorks.empty()) lightWorks += ",\n        "sv;
        else lightWorks += "    "sv;
        lightWorks += FormatToGlobalBuffer(LIGHT_WORK_TEMPLATE_FMT.size(),
                                           LIGHT_WORK_TEMPLATE_FMT,
                                           l.typeName, t.typeName);
    }
    // CAMERA WORKS
    for(const auto& c : lp.cams)
    for(const auto& t : lp.trans)
    {
        if(LookFilterAndSkip(c.transFilters, t)) continue;

        if(!camWorks.empty()) camWorks += ",\n        "sv;
        else camWorks += "    "sv;
        camWorks += FormatToGlobalBuffer(CAM_WORK_TEMPLATE_FMT.size(),
                                         CAM_WORK_TEMPLATE_FMT,
                                         c.typeName, t.typeName);
    }
}

void GenRenderWorkList(pmr::string& workList, const LinePack& lp)
{
    static constexpr auto WORK_LIST_FMT = "RendererWorkTypes<{}, {}, {}, {}>"sv;
    std::string buffer;
    for(const auto& r : lp.renders)
    {
        if(!workList.empty()) workList += ",\n    "sv;
        else workList += "    "sv;
        workList += FormatToGlobalBuffer(WORK_LIST_FMT.size(),
                                         WORK_LIST_FMT,
                                         r.typeName,
                                         r.workOverload,
                                         r.lightWorkOverload,
                                         r.cameraWorkOverload);
    }
}

void WriteRequestedTypesFiles(const LinePack& lp,
                             std::filesystem::path outDir,
                             std::string_view hwAccelHeaderGuard)
{
    auto includes = pmr::string(&globalAllocator);
    auto guardedInclude = pmr::string(&globalAllocator);
    auto primNames = pmr::string(&globalAllocator);
    auto matNames = pmr::string(&globalAllocator);
    auto transNames = pmr::string(&globalAllocator);
    auto camNames = pmr::string(&globalAllocator);
    auto medNames = pmr::string(&globalAllocator);
    auto lightNames = pmr::string(&globalAllocator);
    auto metaLightTypePack = pmr::string(&globalAllocator);
    auto accelGroupTypePack = pmr::string(&globalAllocator);
    auto accelWorkTypePack = pmr::string(&globalAllocator);
    std::array<std::string_view, 2> linAccelNamePair;
    std::array<std::string_view, 2> bvhAccelNamePair;
    std::array<std::string_view, 2> hwAccelNamePair;
    //
    auto renderIncludes = pmr::string(&globalAllocator);
    auto rendererNames = pmr::string(&globalAllocator);
    auto renderWorkTypePack = pmr::string(&globalAllocator);
    auto renderLightWorkTypePack = pmr::string(&globalAllocator);
    auto renderCamWorkTypePack = pmr::string(&globalAllocator);
    auto renderWorkList = pmr::string(&globalAllocator);

    auto AppendWithComma = [](pmr::string& s, std::string_view name)
    {
        if(s.empty())   s += "    "sv;
        else            s += ",\n    "sv;
        s += name;
    };
    //
    for(const auto& prim : lp.prims)
        AppendWithComma(primNames, prim.typeName);
    for(const auto& mat : lp.mats)
        AppendWithComma(matNames, mat.typeName);
    for(const auto& trans : lp.trans)
        AppendWithComma(transNames, trans.typeName);
    for(const auto& cam : lp.cams)
        AppendWithComma(camNames, cam.typeName);
    for(const auto& med : lp.meds)
        AppendWithComma(medNames, med.typeName);
    for(const auto& light : lp.lights)
        AppendWithComma(lightNames, light.typeName);
    for(const auto& rend : lp.renders)
        AppendWithComma(rendererNames, rend.typeName);

    FilterAndGenIncludes(includes, renderIncludes, lp);
    GenMetaLightTemplates(metaLightTypePack, lp);
    GenAcceleratorTemplates(accelGroupTypePack, accelWorkTypePack, lp);
    FindAccelNames(linAccelNamePair, bvhAccelNamePair,
                   hwAccelNamePair, guardedInclude, lp);
    GenRenderWorkTemplates(renderWorkTypePack, renderLightWorkTypePack,
                           renderCamWorkTypePack, lp);
    GenRenderWorkList(renderWorkList, lp);

    auto fileStr = FormatToGlobalBuffer(TYPEGEN_HEADER_FILE_TEMPLATE_FMT.size(),
                                        TYPEGEN_HEADER_FILE_TEMPLATE_FMT,
                                        includes,               hwAccelHeaderGuard,
                                        guardedInclude,         primNames,
                                        matNames,               transNames,
                                        camNames,               medNames,
                                        lightNames,             metaLightTypePack,
                                        accelGroupTypePack,     accelWorkTypePack,
                                        linAccelNamePair[0],    linAccelNamePair[1],
                                        bvhAccelNamePair[0],    bvhAccelNamePair[1],
                                        hwAccelHeaderGuard,     hwAccelNamePair[0],
                                        hwAccelNamePair[1]);
    auto reqFilePath = outDir / std::filesystem::path(TYPEGEN_HEADER_FILE);
    ConditionallyWriteToFile(reqFilePath, fileStr);
    //
    fileStr = FormatToGlobalBuffer(TYPEGEN_RENDER_HEADER_FILE_TEMPLATE_FMT.size(),
                                   TYPEGEN_RENDER_HEADER_FILE_TEMPLATE_FMT,
                                   renderIncludes,          renderWorkTypePack,
                                   renderLightWorkTypePack, renderCamWorkTypePack,
                                   rendererNames,           renderWorkList);
    auto rendReqFilePath = outDir / std::filesystem::path(TYPEGEN_RENDER_HEADER_FILE);
    ConditionallyWriteToFile(rendReqFilePath, fileStr);
}

void GenerateKernelInstantiationFiles(const LinePack& lp,
                                      std::filesystem::path outDir,
                                      size_t fileCount)
{
    static constexpr auto WORK_FMT = "MRAY_RENDERER_KERNEL_INSTANTIATE({}, {}, {}, {}, {});\n"sv;
    static constexpr auto LIGHT_WORK_FMT = "MRAY_RENDERER_LIGHT_KERNEL_INSTANTIATE({}, {}, {}, {});\n"sv;
    static constexpr auto CAM_WORK_FMT = "MRAY_RENDERER_CAM_KERNEL_INSTANTIATE({}, {}, {}, {});\n"sv;
    //
    static constexpr auto ACCEL_PRIM_CENTER_FMT = "MRAY_ACCEL_PRIM_CENTER_KERNEL_INSTANTIATE({}, {}, {});\n"sv;
    static constexpr auto ACCEL_PRIM_AABB_FMT = "MRAY_ACCEL_PRIM_AABB_KERNEL_INSTANTIATE({}, {}, {});\n"sv;
    static constexpr auto ACCEL_TRANSFORM_FMT = "MRAY_ACCEL_COMMON_TRANSFORM_KERNEL_INSTANTIATE({});\n"sv;
    static constexpr auto ACCEL_TRANSFORM_AABB_FMT = "MRAY_ACCEL_TRANSFORM_AABB_KERNEL_INSTANTIATE({}, {}, {});\n"sv;
    static constexpr auto ACCEL_RC_LOCAL_FMT = "MRAY_ACCEL_LOCAL_RAY_CAST_KERNEL_INSTANTIATE({}, {}, {});\n"sv;
    static constexpr auto ACCEL_RC_VISIBILITY_FMT = "MRAY_ACCEL_VISIBILITY_RAY_CAST_KERNEL_INSTANTIATE({}, {}, {});\n"sv;
    //
    static constexpr auto CAM_SUBCAMERA_FMT = "MRAY_RAYGEN_SUBCAMERA_KERNEL_INSTANTIATE({}, {});\n"sv;
    static constexpr auto CAM_RAYGEN_FMT = "MRAY_RAYGEN_GENRAYS_KERNEL_INSTANTIATE({}, {});\n"sv;
    static constexpr auto CAM_RAYGEN_STOCHASTIC_FMT = "MRAY_RAYGEN_GENRAYS_STOCHASTIC_KERNEL_INSTANTIATE({}, {}, {});\n"sv;

    namespace pmr = pmr;
    using StringViewVec = pmr::vector<std::string_view>;
    auto workInstantiations = StringViewVec(&globalAllocator);      workInstantiations.reserve(1024);
    auto lightWorkInstantiations = StringViewVec(&globalAllocator); lightWorkInstantiations.reserve(1024);
    auto camWorkInstantiations = StringViewVec(&globalAllocator);   camWorkInstantiations.reserve(1024);
    //
    auto accelInstantiations = StringViewVec(&globalAllocator);     accelInstantiations.reserve(1024);
    //
    auto rayGenInstantiations = StringViewVec(&globalAllocator);    rayGenInstantiations.reserve(1024);
    // Work Gen
    for(const auto& r : lp.renders)
    {
        // SCATTER WORKS
        for(const auto& p : lp.prims)
        {
            if(p.typeName == "PrimGroupEmpty"sv) continue;
            for(const auto& m : lp.mats)
            {
                if(LookFilterAndSkip(p.matFilters, m)) continue;
                if(LookFilterAndSkip(m.primFilters, p)) continue;
                for(const auto& t : lp.trans)
                for(uint32_t i = 0; i < r.workCount; i++)
                {
                    if(LookFilterAndSkip(p.transFilters, t)) continue;
                    if(LookFilterAndSkip(m.transFilters, t)) continue;
                    if(LookFilterAndSkip(t.primFilters, p)) continue;
                    if(LookFilterAndSkip(t.matFilters, m)) continue;

                    auto wILine = FormatToGlobalBuffer(WORK_FMT.size(), WORK_FMT,
                                                       r.typeName, p.typeName,
                                                       m.typeName, t.typeName, i);
                    workInstantiations.emplace_back(wILine);
                }
            }
        }
        // LIGHT WORKS
        for(const auto& l : lp.lights)
        for(const auto& t : lp.trans)
        for(uint32_t i = 0; i < r.lightWorkCount; i++)
        {
            if(LookFilterAndSkip(l.transFilters, t)) continue;

            auto wlLine = FormatToGlobalBuffer(LIGHT_WORK_FMT.size(),
                                                LIGHT_WORK_FMT,
                                                r.typeName, l.typeName,
                                                t.typeName, i);
            lightWorkInstantiations.emplace_back(wlLine);
        }
        // CAMERA WORKS

        for(const auto& c : lp.cams)
        for(const auto& t : lp.trans)
        for(uint32_t i = 0; i < r.camWorkCount; i++)
        {
            if(LookFilterAndSkip(c.transFilters, t)) continue;

            auto wcLine = FormatToGlobalBuffer(CAM_WORK_FMT.size(),
                                               CAM_WORK_FMT,
                                               r.typeName, c.typeName,
                                               t.typeName, i);
            camWorkInstantiations.emplace_back(wcLine);
        }
    }

    // Accel Gen
    for(const auto& a : lp.accels)
    for(const auto& t : lp.trans)
    for(const auto& p : lp.prims)
    {
        if(p.typeName == "PrimGroupEmpty"sv) continue;
        if(LookFilterAndSkip(p.transFilters, t)) continue;
        if(LookFilterAndSkip(t.primFilters, p)) continue;

        auto line = std::string_view();
        // KCGenPrimCenters
        line = FormatToGlobalBuffer(ACCEL_PRIM_CENTER_FMT.size(),
                                    ACCEL_PRIM_CENTER_FMT,
                                    a.groupName, p.typeName,
                                    t.typeName);
        accelInstantiations.emplace_back(line);
        // KCGenPrimAABBs
        line = FormatToGlobalBuffer(ACCEL_PRIM_AABB_FMT.size(),
                                    ACCEL_PRIM_AABB_FMT,
                                    a.groupName, p.typeName,
                                    t.typeName);
        accelInstantiations.emplace_back(line);
        // KCTransformLocallyConstantAABBs
        line = FormatToGlobalBuffer(ACCEL_TRANSFORM_AABB_FMT.size(),
                                    ACCEL_TRANSFORM_AABB_FMT,
                                    a.groupName, p.typeName,
                                    t.typeName);
        accelInstantiations.emplace_back(line);
        // KCLocalRayCast
        line = FormatToGlobalBuffer(ACCEL_RC_LOCAL_FMT.size(),
                                    ACCEL_RC_LOCAL_FMT,
                                    a.groupName, p.typeName,
                                    t.typeName);
        accelInstantiations.emplace_back(line);
        // KCVisibilityRayCast
        line = FormatToGlobalBuffer(ACCEL_RC_VISIBILITY_FMT.size(),
                                    ACCEL_RC_VISIBILITY_FMT,
                                    a.groupName, p.typeName,
                                    t.typeName);
        accelInstantiations.emplace_back(line);
    }
    //
    for(const auto& t : lp.trans)
    {
        // KCGetCommonTransforms
        auto line = FormatToGlobalBuffer(ACCEL_TRANSFORM_FMT.size(),
                                         ACCEL_TRANSFORM_FMT, t.typeName);
        accelInstantiations.emplace_back(line);
    }
    // Camera Raygen Work
    for(const auto& c : lp.cams)
    for(const auto& t : lp.trans)
    {
        if(LookFilterAndSkip(c.transFilters, t)) continue;

        auto line = std::string_view();
        // KCGenerateSubCamera
        line = FormatToGlobalBuffer(CAM_SUBCAMERA_FMT.size(), CAM_SUBCAMERA_FMT,
                                    c.typeName, t.typeName);
        rayGenInstantiations.emplace_back(line);
        // KCGenerateCamRays
        line = FormatToGlobalBuffer(CAM_RAYGEN_FMT.size(), CAM_RAYGEN_FMT,
                                    c.typeName, t.typeName);
        rayGenInstantiations.emplace_back(line);
        // KCGenerateCamRaysStochastic
        static constexpr std::array FILTER_NAMES =
        {
            "BoxFilter"sv,
            "TentFilter"sv,
            "GaussianFilter"sv,
            "MitchellNetravaliFilter"sv
        };
        for(const auto& fName : FILTER_NAMES)
        {
            line = FormatToGlobalBuffer(CAM_RAYGEN_STOCHASTIC_FMT.size(),
                                        CAM_RAYGEN_STOCHASTIC_FMT,
                                        c.typeName, t.typeName, fName);
            rayGenInstantiations.emplace_back(line);
        };
    }
    //
    using PerFileVector = pmr::vector<std::array<pmr::string, 5>>;
    PerFileVector perFileLists(fileCount, &globalAllocator);
    // Propogation of our allocator stops due to "std::array"
    // Manually allocate and reserve
    for(auto& arr : perFileLists)
    for(auto& s : arr)
    {
        s = pmr::string(&globalAllocator);
        s.reserve(4096);
    }

    // Split the works
    auto AppendInstantiations = [fileCount](PerFileVector& perFileLists, uint32_t listIndex,
                                            const StringViewVec& instatiationList)
    {
        size_t workInstantiationPerFile = Math::DivideUp(instatiationList.size(), fileCount);
        for(size_t i = 0; i < fileCount; i++)
        for(size_t j = 0; j < workInstantiationPerFile; j++)
        {
            size_t instantiationIndex = i * workInstantiationPerFile + j;
            if(instantiationIndex >= instatiationList.size()) break;

            perFileLists[i][listIndex] += instatiationList[instantiationIndex];
        }
    };
    AppendInstantiations(perFileLists, 0, workInstantiations);
    AppendInstantiations(perFileLists, 1, lightWorkInstantiations);
    AppendInstantiations(perFileLists, 2, camWorkInstantiations);
    AppendInstantiations(perFileLists, 3, accelInstantiations);
    AppendInstantiations(perFileLists, 4, rayGenInstantiations);
    /*for(size_t i = 0; i < workInstantiations.size(); i++)
        perFileLists[i % fileCount][0] += workInstantiations[i];
    for(size_t i = 0; i < lightWorkInstantiations.size(); i++)
        perFileLists[i % fileCount][1] += lightWorkInstantiations[i];
    for(size_t i = 0; i < camWorkInstantiations.size(); i++)
        perFileLists[i % fileCount][2] += camWorkInstantiations[i];

    for(size_t i = 0; i < accelInstantiations.size(); i++)
        perFileLists[i % fileCount][3] += accelInstantiations[i];
    for(size_t i = 0; i < rayGenInstantiations.size(); i++)
        perFileLists[i % fileCount][4] += rayGenInstantiations[i];*/

    for(size_t i = 0; i < size_t(fileCount); i++)
    {
        std::string_view fName = FormatToGlobalBuffer(KERNEL_FILE_FMT.size(),
                                                      KERNEL_FILE_FMT, i + 1);

        if(perFileLists[i][0].empty() &&
           perFileLists[i][1].empty() &&
           perFileLists[i][2].empty() &&
           perFileLists[i][3].empty() &&
           perFileLists[i][4].empty())
        {
            ConditionallyWriteToFile(outDir / fName, {});
            continue;
        }

        auto fContents = FormatToGlobalBuffer(KERNEL_FILE_TEMPLATE.size(),
                                              KERNEL_FILE_TEMPLATE,
                                              perFileLists[i][0],
                                              perFileLists[i][1],
                                              perFileLists[i][2],
                                              perFileLists[i][3],
                                              perFileLists[i][4]);
        ConditionallyWriteToFile(outDir / fName, fContents);
    }
}

int main(int argc, const char* argv[])
{
    Timer t; t.Start();

    uint32_t argcUInt = static_cast<uint32_t>(argc);
    static constexpr uint32_t MAX_ARG_COUNT = 4;
    if(argcUInt != MAX_ARG_COUNT + 1)
    {
        fmt::println(stderr, "Wrong Argument Count({})", argcUInt);
        return 1;
    }
    std::array<std::string_view, MAX_ARG_COUNT> args;
    for(uint32_t i = 0; i < std::min(argcUInt, MAX_ARG_COUNT); i++)
        args[i] = argv[i + 1];

    size_t fileCount = 0;
    if(std::from_chars(args[1].data(), args[1].data() + args[1].size(),
                       fileCount).ec != std::errc())
    {
        fmt::println(stderr, "2nd arg is not a number. ({})", args[1]);
        return 1;
    };
    fileCount = std::max(size_t(1), std::min(fileCount, size_t(std::thread::hardware_concurrency())));

    auto outDir = args[2];
    auto headerGuard = args[3];

    // Don't use data() anywhere else like this, it is UB since string_view
    // is not null terminated!!!!
    // args[0] wraps null terminated c-string so its fine here
    std::ifstream file = std::ifstream(args[0].data(), std::ios::binary);
    if(!file.is_open() || !file)
    {
        fmt::println("Unable to open file \"{}\"", args[0]);
        return 1;
    }

    // Load the data
    auto data = pmr::string(std::istreambuf_iterator<char>(file), {}, &globalAllocator);
    // Parse the data
    LinePack lp;
    ParseTypes(lp, data);

    //
    WriteRequestedTypesFiles(lp, outDir, headerGuard);
    GenerateKernelInstantiationFiles(lp, outDir, fileCount);

    t.Split();
    //fmt::println("Generation took {:f}ms", t.Elapsed<Millisecond>());
    return 0;
}