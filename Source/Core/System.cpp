#include "System.h"
#include <filesystem>
#include <thread>
#include <type_traits>

#include "DataStructures.h"
#include "Error.h"
#include "Error.hpp"

#ifdef MRAY_WINDOWS

    #include <windows.h>

#elif defined MRAY_LINUX

    #include <pthread.h>
    #include <sys/ioctl.h>

#else
#error System preprocessor definition is not set properly! (CMake should have handled this)
#endif

// Static checks of forward declared types
// Common part
static_assert(std::is_same_v<std::thread::native_handle_type, SystemThreadHandle>,
              "Forward declared thread handle type does not "
              "match with actual system's type!");

#ifdef MRAY_WINDOWS

// this is technically HANDLE = HANDLE, but just to be sure,
// compiler shoul've failed while parsing the windows.h
// when "typedef void* HANDLE" mismatched with the actual windows.h's
// declaration (aliasing). Maybe user/maintainer will see this
// and could be able to do an informed action
static_assert(std::is_same_v<HANDLE, SystemSemaphoreHandle>,
              "Forward declared semaphore handle type does not "
              "match with system's type!");
static_assert(std::is_same_v<LPVOID, SystemMemoryHandle>,
              "Forward declared semaphore handle type does not "
              "match with system's type!");

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

// https://stackoverflow.com/questions/23369503/get-size-of-terminal-window-rows-columns
std::array<size_t, 2>
GetTerminalSize()
{
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    int columns, rows;

    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    columns = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
    //
    assert(columns >= 0 && rows >= 0);
    return
    {
        static_cast<size_t>(columns),
        static_cast<size_t>(rows)
    };
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

// https://stackoverflow.com/questions/23369503/get-size-of-terminal-window-rows-columns
std::array<size_t, 2>
GetTerminalSize()
{
    struct winsize w = {};
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    return
    {
        static_cast<size_t>(w.ws_col),
        static_cast<size_t>(w.ws_row)
    };
}

#else
#error System preprocessor definition is not set properly! (CMake should have handled this)
#endif

