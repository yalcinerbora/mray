#include "MRayDataType.h"

MRayDataEnum MRayDataTypeRT::Name() const
{
    return std::visit([](auto&& d) -> MRayDataEnum
    {
        return std::remove_cvref_t<decltype(d)>::Name;
    }, *this);
}

size_t MRayDataTypeRT::Size() const
{
    return std::visit([](auto&& d) -> size_t
    {
        return std::remove_cvref_t<decltype(d)>::Size;
    }, *this);
}

size_t MRayDataTypeRT::Alignment() const
{
    return std::visit([](auto&& d) -> size_t
    {
        return std::remove_cvref_t<decltype(d)>::Alignment;
    }, *this);
}

MRayPixelEnum MRayPixelTypeRT::Name() const
{
    return std::visit([](auto&& d) -> MRayPixelEnum
    {
        return std::remove_cvref_t<decltype(d)>::Name;
    }, *this);
}

size_t MRayPixelTypeRT::ChannelCount() const
{
    return std::visit([](auto&& d) -> size_t
    {
        return std::remove_cvref_t<decltype(d)>::ChannelCount;
    }, *this);
}

size_t MRayPixelTypeRT::PixelSize() const
{
    return std::visit([](auto&& d) -> size_t
    {
        return std::remove_cvref_t<decltype(d)>::PixelSize;
    }, *this);
}

bool MRayPixelTypeRT::IsBlockCompressed() const
{
    return std::visit([](auto&& d) -> bool
    {
        return std::remove_cvref_t<decltype(d)>::IsBCPixel;
    }, *this);
}
