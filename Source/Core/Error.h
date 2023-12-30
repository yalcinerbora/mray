#pragma once

#include <string>
//#include <source_location>
#include "Log.h"

// Very generic error
struct MRayError
{
    public:
    enum Type
    {
        OK,
        ERROR,
    };

    private:
    Type                    type = OK;
    std::string             customInfo;
//    std::source_location    sourceInfo;

    public:
    MRayError(std::string && = "");
        //, std::source_location = std::source_location::current());

    operator bool();

    std::string GetError() const;
    std::string AppendInfo(const std::string&);

};

inline MRayError::MRayError(std::string&& s)
                            //std::source_location loc)
    : type(MRayError::ERROR)
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

inline std::string MRayError::AppendInfo(const std::string& s)
{
    customInfo += s;
}