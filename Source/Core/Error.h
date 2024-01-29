#pragma once

#include <string>
//#include <source_location>
#include "Log.h"

// Very generic error
// Throw it return it, stick in a stew
struct MRayError
{
    public:
    enum Type
    {
        OK,
        HAS_ERROR
    };

    private:
    Type                    type = OK;
    std::string             customInfo;
    // std::source_location crashes the nvcc compiler (at least on CUDA 12.1)
    // so this is commented out will be enabled when it is fixed
    //std::source_location    sourceInfo;

    public:
    MRayError(Type);
    MRayError(std::string && = "");
        //, std::source_location = std::source_location::current());

    operator bool();

    std::string GetError() const;
    void        AppendInfo(const std::string&);

};

inline MRayError::MRayError(Type t)
    : type(t)
{}

inline MRayError::MRayError(std::string&& s)
                            //std::source_location loc)
    : type(MRayError::HAS_ERROR)
    , customInfo(s)
    //, sourceInfo(loc)
{}

inline MRayError::operator bool()
{
    return type == MRayError::OK;
}

inline std::string MRayError::GetError() const
{
    return MRAY_FORMAT("Error ||| {:s}", customInfo);
    //return MRAY_FORMAT("{:s}[{:d}]:[{:d}] -> {:s} ||| {:s}",
    //                   sourceInfo.file_name(), sourceInfo.line(),
    //                   sourceInfo.column(),
    //                   sourceInfo.function_name(), customInfo);
}

inline void MRayError::AppendInfo(const std::string& s)
{
    customInfo += s;
}