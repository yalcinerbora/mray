#pragma once

#include <vector>
#include <string>
#include <memory>

#include "ImageLoaderI.h"

#include "MRayInput/MRayInput.h"

#include <OpenImageIO/imageio.h>

class ImageLoader final : public ImageLoaderI
{
    private:
        static constexpr size_t PARALLEL_EXEC_TRESHOLD = 2048;

        // Conversion Enums
        static Expected<MRayPixelTypeRT>    OIIOImageSpecToPixelFormat(const OIIO::ImageSpec&);
        static Expected<OIIO::ImageSpec>    PixelFormatToOIIOImageSpec(const ImageHeader<2>& header);

    protected:
    public:
        // Constructors & Destructor
                            ImageLoader() = default;
                            ImageLoader(const ImageLoader&) = delete;
        ImageLoader&        operator=(const ImageLoader&) = delete;
                            ~ImageLoader() = default;

        // Interface
        Expected<Image<2>>          ReadImage2D(const std::string& filePath,
                                                const ImageIOFlags = ImageIOFlags()) const override;

        Expected<Image<2>>          ReadImageSubChannel(const std::string& filePath,
                                                        ImageChannelType,
                                                        const ImageIOFlags = ImageIOFlags()) const override;

        Expected<ImageHeader<2>>    ReadImageHeader2D(const std::string& filePath,
                                                      const ImageIOFlags = ImageIOFlags()) const override;

        MRayError                   WriteImage2D(const Image<2>&,
                                                 const std::string& filePath,
                                                 ImageType extension,
                                                 const ImageIOFlags = ImageIOFlags()) const override;
};