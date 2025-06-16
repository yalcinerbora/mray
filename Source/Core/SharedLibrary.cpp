#include <cassert>

#include "SharedLibrary.h"
#include "System.h"
#include "Filesystem.h"

// Env Headers
#if defined MRAY_WINDOWS

    #include <windows.h>
    #include <strsafe.h>

#elif defined MRAY_LINUX

    #include <dlfcn.h>

#endif

#if defined MRAY_WINDOWS

static std::string FormatErrorWin32()
{
    DWORD errorMessageID = GetLastError();

    LPSTR messageBuffer = nullptr;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                                 FORMAT_MESSAGE_FROM_SYSTEM |
                                 FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, errorMessageID,
                                 MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                 reinterpret_cast<LPSTR>(&messageBuffer),
                                 0, NULL);
    // Get the buffer
    std::string message(messageBuffer, size);
    LocalFree(messageBuffer);
    return message;
}

static std::wstring ConvertWCharWin32(const std::string& unicodeStr)
{

        const size_t length = unicodeStr.length();
        const DWORD kFlags = MB_ERR_INVALID_CHARS;

        // Query string size
        const int utf16Length = ::MultiByteToWideChar(
            CP_UTF8,                    // Source string is in UTF-8
            kFlags,                     // Conversion flags
            unicodeStr.data(),          // Source UTF-8 string pointer
            static_cast<int>(length),   // Length of the source UTF-8 string, in chars
            nullptr,                    // Unused - no conversion done in this step
            0                           // Request size of destination buffer, in wchar_ts
        );

        std::wstring wString(static_cast<size_t>(utf16Length), L'\0');

        // Convert from UTF-8 to UTF-16
        ::MultiByteToWideChar(
            CP_UTF8,                    // Source string is in UTF-8
            kFlags,                     // Conversion flags
            unicodeStr.data(),          // Source UTF-8 string pointer
            static_cast<int>(length),   // Length of source UTF-8 string, in chars
            wString.data(),             // Pointer to destination buffer
            utf16Length                 // Size of destination buffer, in wchar_ts
        );
        return wString;
}

#endif

void* SharedLibrary::GetProcAdressInternal(const std::string& fName) const
{
    #ifdef MRAY_WINDOWS
        FARPROC proc = GetProcAddress(std::bit_cast<HINSTANCE>(libHandle), fName.c_str());
        if(proc == nullptr)
        {
            throw MRayError("{}", FormatErrorWin32());
        }
        return std::bit_cast<void*>(proc);
    #elif defined MRAY_LINUX
        void* result = dlsym(libHandle, fName.c_str());
        if(result == nullptr)
            throw MRayError("{}", dlerror());
        return result;
    #endif
}

bool SharedLibrary::CheckLibExists(std::string_view libName)
{
    std::string fullName;
    #if defined MRAY_WINDOWS
        fullName = std::string(libName) + WinDLLExt;
    #elif defined MRAY_LINUX
        fullName = "lib" + std::string(libName) + LinuxDLLExt;
    #endif

    fullName = Filesystem::RelativePathToAbsolute(fullName, GetProcessPath());
    return std::filesystem::exists(std::filesystem::path(fullName));
}

SharedLibrary::SharedLibrary(const std::string& libName)
{
    std::string potentialError;
    std::string libWithExt = libName;
    #if defined MRAY_WINDOWS
        libWithExt += WinDLLExt;
        libHandle = std::bit_cast<void*>(LoadLibrary(ConvertWCharWin32(libWithExt).c_str()));
        if(libHandle == nullptr)
        {
            potentialError = FormatErrorWin32();
        }
    #elif defined MRAY_LINUX
        libWithExt = "lib";
        libWithExt += libName;
        libWithExt += LinuxDLLExt;
        libHandle = dlopen(libWithExt.c_str(), RTLD_LAZY);
        if(libHandle == nullptr)
            potentialError = dlerror();
    #endif

    if(libHandle == nullptr)
        throw MRayError("[DLLError] ({}) {}",
                        libName, potentialError);
}

SharedLibrary::~SharedLibrary()
{
    #if defined MRAY_WINDOWS
        if(libHandle != nullptr) FreeLibrary(std::bit_cast<HINSTANCE>(libHandle));
    #elif defined MRAY_LINUX
        if(libHandle != nullptr) dlclose(libHandle);
    #endif
}

SharedLibrary::SharedLibrary(SharedLibrary&& other) noexcept
    : libHandle(other.libHandle)
{
    other.libHandle = nullptr;
}

SharedLibrary& SharedLibrary::operator=(SharedLibrary&& other) noexcept
{
    assert(this != &other);

    #if defined MRAY_WINDOWS
        if(libHandle != nullptr) FreeLibrary(std::bit_cast<HINSTANCE>(libHandle));
    #elif defined MRAY_LINUX
        if(libHandle != nullptr) dlclose(libHandle);
    #endif

    libHandle = other.libHandle;
    other.libHandle = nullptr;
    return *this;
}