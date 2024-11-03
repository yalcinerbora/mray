#include "ImageLoader.h"

#include <execution>
#include <algorithm>
#include <array>
#include <filesystem>

#include "Core/MRayDataType.h"

bool ProgressCallbackOIIO(void* progressPercentData, float portion)
{
    auto floatPtr = static_cast<float*>(progressPercentData);
    std::atomic_ref<float>(*floatPtr).store(portion);
    return false;
};

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
    constexpr auto DDSGenFunction = &GenerateType<ImageFileI, ImageFileDDS,
                                                  const std::string&, ImageSubChannelType,
                                                  ImageIOFlags>;
    genMap.emplace(".dds"sv, DDSGenFunction);

    constexpr auto OIIOGenFunction = &GenerateType<ImageFileI, ImageFileOIIO,
                                                   const std::string&, ImageSubChannelType,
                                                   ImageIOFlags>;
    auto loc = genMap.emplace("default"sv, OIIOGenFunction).first;
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
        return defaultGenerator(filePath,
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
                                  ImageIOFlags flags,
                                  float* progressPercentData) const
{
    using enum ImageIOFlags::F;
    const auto& extE = ImageTypeToExtension(extension);
    if(!extE.has_value()) return extE.error();

    std::string_view ext = extE.value();
    std::string fullPath = filePath + std::string(ext);
    auto out = OIIO::ImageOutput::create(fullPath);

    // Output spec
    Expected<OIIO::ImageSpec> outSpecE =
        ImageFileOIIO::PixelFormatToOIIO(imgIn.header.pixelType,
                                         Vector2ui(imgIn.header.dimensions));
    if(!outSpecE.has_value()) return outSpecE.error();
    const OIIO::ImageSpec& outSpec = outSpecE.value();

    // Input spec
    Expected<OIIO::ImageSpec> inSpecE =
        ImageFileOIIO::PixelFormatToOIIO(imgIn.inputType,
                                         Vector2ui(imgIn.header.dimensions));
    if(!inSpecE.has_value()) return inSpecE.error();
    const OIIO::ImageSpec& inSpec = inSpecE.value();

    OIIO::stride_t xStride = static_cast<OIIO::stride_t>(imgIn.inputType.PixelSize());
    OIIO::stride_t yStride = static_cast<OIIO::stride_t>(inSpec.scanline_bytes());
    const Byte* dataStart = imgIn.pixels.data();
    if(!flags[FLIP_Y_COORDINATE])
    {
        yStride = -yStride;
        dataStart += (imgIn.header.dimensions[1] - 1) * inSpec.scanline_bytes();
    }

    OIIO::ProgressCallback callback = nullptr;
    void* progressPercentDataVoid = nullptr;
    if(progressPercentData)
    {
        callback = ProgressCallbackOIIO;
        progressPercentDataVoid = progressPercentData;
    }

    // TODO: properly write an error check/out code for these.
    if(!out->open(fullPath, outSpec))
        return MRayError("OIIO Error ({})", out->geterror());
    if(!out->write_image(outSpec.format, dataStart, xStride, yStride,
                         OIIO::AutoStride, callback,
                         progressPercentDataVoid))
        return MRayError("OIIO Error ({})", out->geterror());
    if(!out->close())
        return MRayError("OIIO Error ({})", out->geterror());

    return MRayError::OK;
}

