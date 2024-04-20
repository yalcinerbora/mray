#pragma once

#include <vector>
#include <string>
#include <memory>

#include "ImageLoaderI.h"

#include "TransientPool/TransientPool.h"

#include <OpenImageIO/imageio.h>

using ColorSpacePack = std::pair<Float, MRayColorSpaceEnum>;

class ImageLoader final : public ImageLoaderI
{
    private:
        static constexpr size_t PARALLEL_EXEC_TRESHOLD = 2048;

        // Conversion Enums
        static Expected<MRayPixelTypeRT>    PixelFormatToMRay(const OIIO::ImageSpec&);
        static Expected<OIIO::ImageSpec>    PixelFormatToOIIO(const ImageHeader<2>& header);

        static Expected<std::string>        ColorSpaceToOIIO(const ColorSpacePack&);
        static Expected<ColorSpacePack>     ColorSpaceToMRay(const std::string&);

        static Expected<std::string_view>   ImageTypeToExtension(ImageType);

        Expected<ImageHeader<2>>            ReadImageHeaderInternal(const OIIO::ImageInput::unique_ptr& inFile,
                                                                    const std::string& filePath,
                                                                    ImageIOFlags flags) const;

    protected:
    public:
        // Constructors & Destructor
                            ImageLoader() = default;
                            ImageLoader(const ImageLoader&) = delete;
        ImageLoader&        operator=(const ImageLoader&) = delete;
                            ~ImageLoader() = default;

        // Interface
        Expected<Image<2>>          ReadImage2D(const std::string& filePath,
                                                ImageIOFlags = ImageIOFlags()) const override;

        Expected<Image<2>>          ReadImageSubChannel(const std::string& filePath,
                                                        ImageChannelType,
                                                        ImageIOFlags = ImageIOFlags()) const override;

        Expected<ImageHeader<2>>    ReadImageHeader2D(const std::string& filePath,
                                                      ImageIOFlags = ImageIOFlags()) const override;

        MRayError                   WriteImage2D(const WriteImage<2>&,
                                                 const std::string& filePath,
                                                 ImageType extension,
                                                 ImageIOFlags = ImageIOFlags()) const override;
};