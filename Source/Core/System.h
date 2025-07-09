#pragma once

#include <memory>
#include <string>

#ifdef MRAY_WINDOWS

    // Clang on Windows ignores __declspec(dllimport)
    #ifdef MRAY_CLANG
        #define MRAY_DLL_IMPORT
    #else
        #define MRAY_DLL_IMPORT __declspec(dllimport)
    #endif

    // This is fine?
    #define MRAY_DLL_EXPORT __declspec(dllexport)

    static constexpr bool MRAY_IS_ON_WINDOWS    = true;
    static constexpr bool MRAY_IS_ON_LINUX      = false;

    // TODO: Try to skip loading entire windows.h for the handle
    // This is not good, but windows prob will not change the def
    // of handle probably ever. :)
    typedef void* HANDLE;

    using SystemSemaphoreHandle = HANDLE;
    using SystemMemoryHandle = HANDLE;
    using SystemThreadHandle = HANDLE;

    #define MRAY_RESTRICT __restrict

#elif defined MRAY_LINUX

    #define MRAY_DLL_IMPORT
    #define MRAY_DLL_EXPORT

    static constexpr bool MRAY_IS_ON_WINDOWS    = false;
    static constexpr bool MRAY_IS_ON_LINUX      = true;

    using SystemThreadHandle = unsigned long int;
    using SystemSemaphoreHandle = int;
    using SystemMemoryHandle = int;

    #define MRAY_RESTRICT __restrict

#else
    #error System preprocessor definition is not set properly! (CMake should have handled this)
#endif

//
std::string             GetProcessPath();
bool                    EnableVTMode();
void                    RenameThread(SystemThreadHandle,
                                 const std::string& name);
SystemThreadHandle      GetCurrentThreadHandle();
std::array<size_t, 2>   GetTerminalSize();

void*   AlignedAlloc(size_t size, size_t alignment);
void*   AlignedRealloc(void* ptr, size_t size, size_t alignment);
void    AlignedFree(void* ptr, size_t size, size_t alignment);

// Loaded class from a shared library
// Destructor may be from the other side of DLL boundary.
//
// So it is just aliased unique_ptr
//template <class T>
//using ObjGeneratorFunc = T* (*)();
template <class T, class... Args>
using ObjGeneratorFuncArgs = T* (*)(Args...);

template <class T>
using ObjDestroyerFunc = void(*)(T*);

template <class T>
using SharedLibPtr = std::unique_ptr<T, ObjDestroyerFunc<T>>;
