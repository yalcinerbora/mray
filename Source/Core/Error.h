#pragma once

#include <string>
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
    Type        type = OK;
    std::string customInfo;

    public:
                MRayError(Type = Type::OK);
    // This pattern emerges, so added as a template
    template<class... Args>
                MRayError(fmt::format_string<Args...> fstr, Args&&... args);

    explicit operator bool();

    std::string GetError() const;
    void        AppendInfo(const std::string&);
};

inline MRayError::MRayError(Type t)
    : type(t)
{}

template<class... Args>
inline MRayError::MRayError(fmt::format_string<Args...> fstr, Args&&... args)
    : type(MRayError::HAS_ERROR)
    , customInfo(MRAY_FORMAT(fstr, std::forward<Args>(args)...))
{}

inline MRayError::operator bool()
{
    return type != MRayError::OK;
}

inline std::string MRayError::GetError() const
{
    return MRAY_FORMAT("|| {:s}", customInfo);
}

inline void MRayError::AppendInfo(const std::string& s)
{
    customInfo += s;
}