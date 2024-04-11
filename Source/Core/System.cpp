#include "System.h"
#include <filesystem>

#include "Core/Error.h"

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

#elif defined MRAY_LINUX

std::string GetProcessPath()
{
    return std::filesystem::canonical("/proc/self/exe").parent_path().string();
}

#else
#error System preprocessor definition is not set properly! (CMake should have handled this)
#endif