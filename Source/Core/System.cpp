#include "System.h"

#include <cstring>
#include <filesystem>
#include <thread>
#include <type_traits>

#include "Math.h"

#ifdef MRAY_WINDOWS

    #include <windows.h>

#elif defined MRAY_LINUX

    #include <malloc.h>
    #include <pthread.h>
    #include <sys/types.h>
    #include <sys/ioctl.h>

#else
#error System preprocessor definition is not set properly! (CMake should have handled this)
#endif

#ifdef MRAY_DEBUG
    #include "BitFunctions.h"
#endif

// Static checks of forward declared types
// Common part
static_assert(std::is_same_v<std::thread::native_handle_type, SystemThreadHandle>,
              "Forward declared thread handle type does not "
              "match with actual system's type!");

#ifdef MRAY_WINDOWS

// this is technically HANDLE = HANDLE, but just to be sure,
// compiler should've failed while parsing the windows.h
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

std::string_view        GetThreadName(SystemThreadHandle);

SystemThreadHandle GetCurrentThreadHandle()
{
    return GetCurrentThread();
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

void* AlignedAlloc(size_t size, size_t alignment)
{
    // Windows is hipster as always
    // does not have "std::aligned_alloc"
    // but have its own "_aligned_malloc" so using it.
    // To confuse it is also has its parameters swapped :)
    assert(Bit::PopC(alignment) == 1);
    size_t alignedSize = Math::NextMultiple(size, alignment);
    return _aligned_malloc(alignedSize, align);
}

void* AlignedRealloc(void* ptr, size_t size, size_t alignment)
{
    assert(Bit::PopC(alignment) == 1);
    size_t alignedSize = Math::NextMultiple(size, alignment);
    return _aligned_realloc(ptr, alignedSize, align);
}

void AlignedFree(void* ptr, size_t, size_t)
{
    _aligned_free(ptr);
}

#elif defined MRAY_LINUX

void RenameThread(std::thread::native_handle_type t, const std::string& name)
{
    static_assert(std::is_same_v<std::thread::native_handle_type, pthread_t>);
    pthread_setname_np(t, name.c_str());
}

SystemThreadHandle GetCurrentThreadHandle()
{
    return SystemThreadHandle(pthread_self());
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

void* AlignedAlloc(size_t size, size_t alignment)
{
    assert(Bit::PopC(alignment) == 1);
    size_t alignedSize = Math::NextMultiple(size, alignment);
    return std::aligned_alloc(alignment, alignedSize);
}

void* AlignedRealloc(void* ptr, size_t size, size_t alignment)
{
    // TODO: Is there really no aligned_realloc on Linux??
    //https://stackoverflow.com/questions/64884745/is-there-a-linux-equivalent-of-aligned-realloc
    assert(Bit::PopC(alignment) == 1);
    size_t alignedSize = Math::NextMultiple(size, alignment);
    auto oldSize = malloc_usable_size(ptr);
    void* newPtr = std::aligned_alloc(alignment, alignedSize);
    std::memcpy(newPtr, ptr, std::min(oldSize, alignedSize));
    std::free(ptr);
    return newPtr;
}

void AlignedFree(void* ptr, size_t, size_t)
{
    std::free(ptr);
}

#else
#error System preprocessor definition is not set properly! (CMake should have handled this)
#endif

