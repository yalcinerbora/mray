#pragma once

#include <memory>
#include <string>

std::string GetProcessPath();

#ifdef MRAY_WINDOWS

    #define MRAY_DLL_IMPORT __declspec(dllimport)
    #define MRAY_DLL_EXPORT __declspec(dllexport)

    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>

    static inline bool EnableVTMode()
    {
        auto SetVT = [](DWORD outputHandle) -> bool
        {
            // Set output mode to handle virtual terminal sequences
            HANDLE hOut = GetStdHandle(outputHandle);
            if(hOut == INVALID_HANDLE_VALUE)
                return false;

            DWORD dwMode = 0;
            if(!GetConsoleMode(hOut, &dwMode))
                return false;

            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            if(!SetConsoleMode(hOut, dwMode))
                return false;
            return true;
        };

        return (SetVT(STD_OUTPUT_HANDLE) &&
                SetVT(STD_ERROR_HANDLE));
    }

    static constexpr bool MRAY_IS_ON_WINDOWS    = true;
    static constexpr bool MRAY_IS_ON_LINUX      = false;

    using SystemMemoryHandle = HANDLE;

#elif defined MRAY_LINUX

    #define MRAY_DLL_IMPORT
    #define MRAY_DLL_EXPORT

    // Linux has already virtual terminal processing?
    static inline bool EnableVTMode()
    {
        return true;
    }

    static constexpr bool MRAY_IS_ON_WINDOWS    = false;
    static constexpr bool MRAY_IS_ON_LINUX      = true;

    using SystemMemoryHandle = int;

#else
    #error System preprocessor definition is not set properly! (CMake should have handled this)
#endif

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
