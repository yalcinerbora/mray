#pragma once

/**

Functionality to Load DLLs or SOs

*/

#include "Error.h"
#include "System.h"

struct SharedLibArgs
{
    std::string         mangledConstructorName = "\0";
    std::string         mangledDestructorName = "\0";

    bool                operator<(const SharedLibArgs& s) const;
};

inline bool SharedLibArgs::operator<(const SharedLibArgs& s) const
{
    std::less<std::string> less;
    return less(mangledConstructorName, s.mangledConstructorName);
}

class SharedLibrary
{
    private:
        static constexpr const char* WinDLLExt      = ".dll";
        static constexpr const char* LinuxDLLExt    = ".so";

        // Props
        void*               libHandle;

        // Internal
        const void*         GetProcAdressInternal(const std::string& fName) const;

    protected:
    public:
        // Constructors & Destructor
                            SharedLibrary(const std::string& libName);
                            SharedLibrary(const SharedLibrary&) = delete;
                            SharedLibrary(SharedLibrary&&) noexcept;
        SharedLibrary&      operator=(const SharedLibrary&) = delete;
        SharedLibrary&      operator=(SharedLibrary&&) noexcept;
                            ~SharedLibrary();

        template <class T, class... Args>
        MRayError           GenerateObjectWithArgs(SharedLibPtr<T>&,
                                                   const SharedLibArgs& mangledNames,
                                                   Args&&...) const;
};

template <class T, class... Args>
MRayError SharedLibrary::GenerateObjectWithArgs(SharedLibPtr<T>& ptr,
                                                const SharedLibArgs& mangledNames,
                                                Args&&... args) const
{
    MRayError err("[DLLError] Exported function name is not found");
    ObjGeneratorFuncArgs<T, Args&&...> genFunc = reinterpret_cast<ObjGeneratorFuncArgs<T, Args&&...>>(GetProcAdressInternal(mangledNames.mangledConstructorName));
    if(!genFunc) return err;
    ObjDestroyerFunc<T> destFunc = reinterpret_cast<ObjDestroyerFunc<T>>(GetProcAdressInternal(mangledNames.mangledDestructorName));
    if(!destFunc) return err;
    ptr = SharedLibPtr<T>(genFunc(std::forward<Args&&>(args)...), destFunc);
    return MRayError();
}

