
# Using this tutorial to implement this
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/tutorials/How-To-Find-Libraries
#
# and looking to here
# https://github.com/NVIDIA/OptiX_Apps
# and lookig in the optix samples

# Check if Optix Path is coming from command line
set(OPTIX_INSTALL_DIR $ENV{OPTIX_INSTALL_DIR})

# If not try to find the Optix

# This is a fresh renderer, we start from 8.0.0 and above
if(WIN32)
    # Default Installation Locations
    set(OPTIX_POTENTIAL_PATH_LIST
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0"
        )
else()
    set(OPTIX_POTENTIAL_PATH_LIST
            "/usr/local/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64"
            "/usr/local/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64"
            "/usr/local/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64"

            "/opt/nvidia/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64"
            "/opt/nvidia/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64"
            "/opt/nvidia/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64"

            "~/NVIDIA-Optix-SDK-9.0.0-linux64"
            "~/NVIDIA-Optix-SDK-8.1.0-linux64"
            "~/NVIDIA-Optix-SDK-8.0.0-linux64"
        )
endif()

# Try to find optix header
if("${OPTIX_INSTALL_DIR}" STREQUAL "")
    find_path(OPTIX_INSTALL_DIR
        NAME include/optix.h
        PATHS ${OPTIX_POTENTIAL_PATH_LIST}
    )
endif()

# Set Include Folder
set(OPTIX_INCLUDE_DIR ${OPTIX_INSTALL_DIR}/include)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OPTIX DEFAULT_MSG OPTIX_INSTALL_DIR OPTIX_INCLUDE_DIR)
