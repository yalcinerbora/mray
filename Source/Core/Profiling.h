#pragma once

#ifdef MRAY_ENABLE_TRACY

    #define TRACY_ENABLE
    #include <tracy/tracy/TracyC.h>
    #include <string_view>
    #include <cassert>
    #include <limits>

    #include "BitFunctions.h"
    #include "ColorFunctions.h"
    #include "SharedLibrary.h"
#endif

#include <source_location>

#ifndef MRAY_ENABLE_TRACY

class ProfilerAnnotation
{
    using SourceLoc = std::source_location;

    public:
    class Scope
    {
        private:
        // Constructors
                Scope(const ProfilerAnnotation&) {}
        public:
        // Constructors & Destructor
                Scope(const Scope&) = delete;
                Scope(Scope&&) = delete;
        Scope&  operator=(const Scope&) = delete;
        Scope&  operator=(Scope&&) = delete;
                ~Scope() = default;
    };
    public:
    // Constructors & Destructor
                        ProfilerAnnotation(std::string_view,
                                           const SourceLoc& = SourceLoc::current()) {}
                        ProfilerAnnotation(const ProfilerAnnotation&) = delete;
                        ProfilerAnnotation(ProfilerAnnotation&&) = delete;
    ProfilerAnnotation& operator=(const ProfilerAnnotation&) = delete;
    ProfilerAnnotation& operator=(ProfilerAnnotation&&) = delete;

    [[nodiscard]]
    Scope               AnnotateScope() const {};
};

class ProfilerDLL
{
    public:
                    ProfilerDLL(Optional<bool> = std::nullopt) {}
                    ProfilerDLL(const ProfilerDLL&) = delete;
                    ProfilerDLL(ProfilerDLL&&) = delete;
    ProfilerDLL&    operator=(const ProfilerDLL&) = delete;
    ProfilerDLL&    operator=(ProfilerDLL&&) = delete;
                    ~ProfilerDLL() = default;
};

#else

class ProfilerAnnotation
{
    using TracySRData = ___tracy_source_location_data;
    using SourceLoc = std::source_location;

    public:
    class Scope
    {
        friend ProfilerAnnotation;

        private:
        const ProfilerAnnotation&   a;
        TracyCZoneCtx               ctx;

        protected:
        // Constructors
                Scope(const ProfilerAnnotation&);
        public:
        // Constructors & Destructor
                Scope(const Scope&) = delete;
                Scope(Scope&&) = delete;
        Scope&  operator=(const Scope&) = delete;
        Scope&  operator=(Scope&&) = delete;
                ~Scope();
    };

    private:
    std::string_view    name;
    TracySRData         srcLocData;
    bool                runtimeFlag;
    public:
    // Constructors & Destructor
                        ProfilerAnnotation(std::string_view name,
                                           const SourceLoc& s = SourceLoc::current());
                        ProfilerAnnotation(const ProfilerAnnotation&) = delete;
                        ProfilerAnnotation(ProfilerAnnotation&&) = delete;
    ProfilerAnnotation& operator=(const ProfilerAnnotation&) = delete;
    ProfilerAnnotation& operator=(ProfilerAnnotation&&) = delete;

    [[nodiscard]]
    Scope               AnnotateScope() const;
};

class ProfilerDLL
{
    public:
    ProfilerDLL(Optional<bool> activate = std::nullopt);
    ProfilerDLL(const ProfilerDLL&) = delete;
    ProfilerDLL(ProfilerDLL&&) = delete;
    ProfilerDLL& operator=(const ProfilerDLL&) = delete;
    ProfilerDLL& operator=(ProfilerDLL&&) = delete;
    ~ProfilerDLL();

    static bool IsActive();
};

inline
ProfilerAnnotation::Scope::Scope(const ProfilerAnnotation& annotation)
    : a(annotation)
{
    if(!___tracy_profiler_started()) return;

    ctx = ___tracy_emit_zone_begin_callstack(&a.srcLocData, TRACY_CALLSTACK, true);
}

inline
ProfilerAnnotation::Scope::~Scope()
{
    if(!___tracy_profiler_started()) return;

    ___tracy_emit_zone_end(ctx);
}

inline
ProfilerAnnotation::ProfilerAnnotation(std::string_view name,
                                       const SourceLoc& s)
    : name(name)
    , srcLocData
    {
        .name = name.data(),
        .function = s.function_name(),
        .file = s.file_name(),
        .line = uint32_t(s.line()),
        .color = std::numeric_limits<uint32_t>::max()
    }
{
    if(!___tracy_profiler_started()) return;

    assert(name.data()[name.size()] == '\0' &&
           "Annotation name string_view must be a "
           "null-terminated string!");

    uint64_t n = std::hash<std::string_view>{}(name);
    uint32_t n32 = uint32_t(Bit::FetchSubPortion(n, {0, 32}) ^
                            Bit::FetchSubPortion(n, {32, 64}));
    auto color = Color::RandomColorRGB(n32);
    using NormConversion::ToUNorm;
    uint32_t colorPack = Bit::Compose<8, 8, 8, 8>
    (
        uint32_t(ToUNorm<uint8_t>(color[0])),
        uint32_t(ToUNorm<uint8_t>(color[1])),
        uint32_t(ToUNorm<uint8_t>(color[2])),
        std::numeric_limits<uint8_t>::max()
    );
    srcLocData.color = colorPack;
}

inline
typename ProfilerAnnotation::Scope
ProfilerAnnotation::AnnotateScope() const
{
    return Scope(*this);
}

inline
ProfilerDLL::ProfilerDLL(Optional<bool> activate)
{
    // TODO: This function will be loaded from the DLL,
    // so it is fine for this function to be inline?
    if(activate && *activate)
        ___tracy_startup_profiler();
}

inline
ProfilerDLL::~ProfilerDLL()
{
    if(!___tracy_profiler_started()) return;

    MRAY_LOG("Shutting down the profiler...");
    ___tracy_shutdown_profiler();
    MRAY_LOG("Profiler shut down!");
}

inline
bool ProfilerDLL::IsActive()
{
    return ___tracy_profiler_started();
}

#endif
