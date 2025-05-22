#include "Error.h"

void MRayError::AppendInfo(const std::string& s)
{
    if(customInfo.empty())
        customInfo = MRAY_FORMAT("{:s}", s);
    else
        customInfo += MRAY_FORMAT("|| {:s}", s);
}

MRayError::MRayError(Type t)
    : type(t)
{}

MRayError::MRayError(std::string_view sv)
    : type(MRayError::HAS_ERROR)
    , customInfo(sv)
{}

MRayError::operator bool() const
{
    return type != MRayError::OK;
}

MRayError::operator bool()
{
    return type != MRayError::OK;
}

std::string MRayError::GetError() const
{
    return customInfo;
}

void ErrorList::AddException(MRayError&& err)
{
    size_t location = size.fetch_add(1);
    // If too many exceptions skip it
    if(location < MaxExceptionSize)
        exceptions[location] = std::move(err);
}
