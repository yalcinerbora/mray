#include "System.h"
#include <filesystem>

#include "DataStructures.h"
#include "Error.h"

#ifdef MRAY_WINDOWS

    #include <windows.h>

#elif defined MRAY_LINUX

    #include<pthread.h>

#else
#error System preprocessor definition is not set properly! (CMake should have handled this)
#endif

#ifdef MRAY_WINDOWS

std::string GetProcessPath()
{
    // TODO: WTF is this api...
    static constexpr size_t MAX_EXEC_PATH_SIZE = 512;
    std::string execPath(MAX_EXEC_PATH_SIZE, '\0');
    DWORD result = GetModuleFileNameA(NULL, execPath.data(),
                                      MAX_EXEC_PATH_SIZE);
    size_t l = strnlen(execPath.data(), MAX_EXEC_PATH_SIZE);
    execPath.resize(l, '\0');

    if(result == ERROR_INSUFFICIENT_BUFFER)
        throw MRayError("Executable path is too long, more than {}",
                        MAX_EXEC_PATH_SIZE);

    return std::filesystem::canonical(execPath).parent_path().string();
}

void RenameThread(std::thread::native_handle_type t, const std::string& name)
{
    static constexpr size_t MAX_CHARS = 128;
    StaticVector<wchar_t, MAX_CHARS> wideStr;
    wideStr.resize(name.size() + 1);
    size_t totalConv = 0;
    mbstowcs_s(&totalConv, wideStr.data(),
               wideStr.size(),
               name.c_str(), MAX_CHARS);
    assert(totalConv == wideStr.size());
    SetThreadDescription(t, wideStr.data());
}

bool EnableVTMode()
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

#elif defined MRAY_LINUX

void RenameThread(std::thread::native_handle_type t, const std::string& name)
{
    pthread_setname_np(t, name.c_str());
}

std::string GetProcessPath()
{
    return std::filesystem::canonical("/proc/self/exe").parent_path().string();
}

bool EnableVTMode()
{
    return true;
}

#else
#error System preprocessor definition is not set properly! (CMake should have handled this)
#endif