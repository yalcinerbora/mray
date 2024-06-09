#pragma once

#include <vector>
#include <string>
#include <memory>

#include "ImageLoaderI.h"

#include "TransientPool/TransientPool.h"

#include "Core/TypeGenFunction.h"

#include <OpenImageIO/imageio.h>

class ImageFileOIIO : public ImageFileBase
{
    private:
    OIIO::ImageInput::unique_ptr        oiioFile;
    ImageHeader                         header;

    public:
    // Conversion Enums
    static Expected<MRayPixelTypeRT>    PixelFormatToMRay(const OIIO::ImageSpec&);
    static Expected<OIIO::ImageSpec>    PixelFormatToOIIO(const ImageHeader& header);

    static Expected<std::string>        ColorSpaceToOIIO(const ColorSpacePack&);
    static Expected<ColorSpacePack>     ColorSpaceToMRay(const std::string&);

    public:
    // Constructors & Destructor
                            ImageFileOIIO(const std::string& filePath,
                                          ImageSubChannelType subChannels,
                                          ImageIOFlags flags);

    Expected<ImageHeader>   ReadHeader() override;
    Expected<Image>         ReadImage() override;
};

class ImageFileDDS : public ImageFileBase
{
    private:
    std::ifstream           ddsFile;
    ImageHeader             header;
    bool                    headerIsRead = false;

    public:
    // Constructors & Destructor
                            ImageFileDDS(const std::string& filePath,
                                          ImageSubChannelType subChannels,
                                          ImageIOFlags flags);

    Expected<ImageHeader>   ReadHeader() override;
    Expected<Image>         ReadImage() override;
};

using ImageFileGen = GeneratorFuncType<ImageFileI, const std::string&,
                                       ImageSubChannelType, ImageIOFlags>;
using ImageFileGeneratorMap = std::map<std::string_view, ImageFileGen>;

class ImageLoader final : public ImageLoaderI
{
    private:
    static Expected<std::string_view>   ImageTypeToExtension(ImageType);

    private:
    ImageFileGeneratorMap   genMap;
    ImageFileGen            defaultGenerator;

    public:
        // Constructors & Destructor
                        ImageLoader(bool enableMT = false);
                        ImageLoader(const ImageLoader&) = delete;
        ImageLoader&    operator=(const ImageLoader&) = delete;
                        ~ImageLoader() = default;

        // Interface
        Expected<ImageFilePtr>  OpenFile(const std::string& filePath,
                                                         ImageSubChannelType subChannels = ImageSubChannelType::ALL,
                                                         ImageIOFlags flags = ImageIOFlags()) const override;

        MRayError               WriteImage(const WriteImageParams&,
                                           const std::string& filePath,
                                           ImageType extension,
                                           ImageIOFlags = ImageIOFlags()) const override;
};