#include "ImageLoader.h"
#include "Core/MRayDataType.h"

#include <execution>
#include <algorithm>
#include <array>
#include <filesystem>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/deepdata.h>

Expected<std::string_view> ImageLoader::ImageTypeToExtension(ImageType type)
{
    using namespace std::literals;
    switch(type)
    {
        case ImageType::PNG: return ".png";
        case ImageType::JPG: return ".jpg";
        case ImageType::BMP: return ".bmp";
        case ImageType::HDR: return ".hdr";
        case ImageType::EXR: return ".exr";
        default: return MRayError("Unknown image type");
    }
}

ImageLoader::ImageLoader(bool enableMT)
{
    if(!enableMT)
    {
        OIIO::attribute("threads", 1);
        OIIO::attribute("exr_threads", 1);
    }
    // Do not use all readers (maybe imp. performance)
    // also it is technically an error, user should be aware of
    // these situations (maybe png file is a jpeg, and it is compressed
    // user may pull their hair out to find the error on the rendering etc.)
    OIIO::attribute("try_all_readers", 0);

    using namespace std::string_view_literals;
    genMap.emplace(".dds"sv, &GenerateType<ImageFileI, ImageFileDDS,
                                           const std::string&, ImageSubChannelType, ImageIOFlags>);
    auto loc =
    genMap.emplace("default"sv, &GenerateType<ImageFileI, ImageFileOIIO,
                                              const std::string&, ImageSubChannelType, ImageIOFlags>).first;
    defaultGenerator = loc->second;
}

Expected<ImageFilePtr> ImageLoader::OpenFile(const std::string& filePath,
                                             ImageSubChannelType subChannels,
                                             ImageIOFlags flags) const
{
    using namespace std::string_view_literals;
    std::filesystem::path fp(filePath);
    std::string fileExt = fp.extension().string();
    auto loc = genMap.find(fileExt);
    ImageFileGen generator = (loc == genMap.cend())
                                ? defaultGenerator
                                : loc->second;

    ImageFilePtr filePtr = generator(filePath,
                                     std::move(subChannels),
                                     std::move(flags));
    if(fileExt != ".dds"sv) return filePtr;

    // If file is dds check the header
    auto headerE = filePtr->ReadHeader();
    // If DDS file is OK but uncompressed delegate to OIIO
    if(headerE.has_error() && !headerE.error())
        return generator(filePath,
                         std::move(subChannels),
                         std::move(flags));
    // DDS has errors return it here
    else if(headerE.has_error())
        return headerE.error();
    // DDS is fine return the file
    else return filePtr;
}

MRayError ImageLoader::WriteImage(const WriteImageParams& imgIn,
                                  const std::string& filePath,
                                  ImageType extension,
                                  ImageIOFlags) const
{
    // TODO: Implement deep writing
    if(imgIn.depth >= 1)
        return MRayError("Deep image writing is currently not implemented");

    const auto& extE = ImageTypeToExtension(extension);
    if(!extE.has_value()) return extE.error();

    std::string_view ext = extE.value();
    std::string fullPath = filePath + std::string(ext);
    auto out = OIIO::ImageOutput::create(fullPath);

    Expected<OIIO::ImageSpec> specE = ImageFileOIIO::PixelFormatToOIIO(imgIn.header);
    if(!specE.has_value()) return specE.error();
    const OIIO::ImageSpec& spec = specE.value();

    OIIO::stride_t scanLineSize = static_cast<OIIO::stride_t>(spec.scanline_bytes());

    const Byte* dataLastElement = imgIn.pixels.data();
    dataLastElement += (imgIn.header.dimensions[1] - 1) * scanLineSize;

    // TODO: properly write an error check/out code for these.
    if(!out->open(fullPath, spec))
        return MRayError("OIIO Error ({})", out->geterror());
    if(!out->write_image(spec.format, dataLastElement, OIIO::AutoStride, -scanLineSize))
        return MRayError("OIIO Error ({})", out->geterror());
    if(!out->close())
        return MRayError("OIIO Error ({})", out->geterror());

    return MRayError::OK;
}

