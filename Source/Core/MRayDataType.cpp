#include "MRayDataType.h"

size_t MRayDataTypeRT::Size() const
{
    return SwitchCase([](auto&& d) -> size_t
    {
        return std::remove_cvref_t<decltype(d)>::Size;
    });
}

size_t MRayDataTypeRT::Alignment() const
{
    return SwitchCase([](auto&& d) -> size_t
    {
        return std::remove_cvref_t<decltype(d)>::Alignment;
    });
}

size_t MRayPixelTypeRT::ChannelCount() const
{
    size_t result;
    SwitchCase([&result](auto&& d) -> bool
    {
        result = std::remove_cvref_t<decltype(d)>::ChannelCount;
        return true;
    });
    return result;
}

bool MRayPixelTypeRT::IsBlockCompressed() const
{
    bool result;
    SwitchCase([&result](auto&& d) -> bool
    {
        result = std::remove_cvref_t<decltype(d)>::IsBCPixel;
        return true;
    });
    return result;
}

bool MRayPixelTypeRT::IsSigned() const
{
    bool result;
    SwitchCase([&result](auto&& d) -> bool
    {
        result = std::remove_cvref_t<decltype(d)>::IsSigned;
        return true;
    });
    return result;
}

size_t MRayPixelTypeRT::PixelSize() const
{
    size_t result;
    SwitchCase([&result](auto&& d) -> bool
    {
        result = std::remove_cvref_t<decltype(d)>::PixelSize;
        return true;
    });
    return result;
}

size_t MRayPixelTypeRT::PaddedPixelSize() const
{
    size_t result;
    SwitchCase([&result](auto&& d) -> bool
    {
        result = std::remove_cvref_t<decltype(d)>::PaddedPixelSize;
        return true;
    });
    return result;
}