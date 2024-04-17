#pragma once

#include <filesystem>

namespace Filesystem
{
    inline std::string RelativePathToAbsolute(std::string_view filePath,
                                              std::string_view relativeLookupPath)
    {
        using namespace std::filesystem;
        // Skip if path is absolute
        if(path(filePath).is_absolute()) return std::string(filePath);
        // Create an absolute path relative to the scene.json file
        path fullPath = path(relativeLookupPath) / path(filePath);
        return absolute(fullPath).generic_string();
    }

};