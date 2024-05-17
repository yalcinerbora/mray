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
    // Constructors & Destructor
                MRayError(Type = Type::OK);
    // This pattern emerges, so added as a template
    template<class... Args>
                MRayError(fmt::format_string<Args...> fstr, Args&&... args);
                MRayError(const MRayError&)     = default;
                MRayError(MRayError&&) noexcept = default;
    MRayError&  operator=(const MRayError&)     = default;
    MRayError&  operator=(MRayError&&) noexcept = default;
                ~MRayError()                    = default;

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
    return customInfo;
}

inline void MRayError::AppendInfo(const std::string& s)
{
    if(customInfo.empty())
        customInfo = MRAY_FORMAT("{:s}", s);
    else
        customInfo += MRAY_FORMAT("|| {:s}", s);
}