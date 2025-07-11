
# https://stackoverflow.com/questions/62553007/cmake-check-python3-and-python-packages-presence
macro(mray_check_pip_package)
    cmake_parse_arguments(MRAY_CHECK_PIP_PACKAGE "" "PACKAGE_NAME" "" ${ARGN})
    execute_process(COMMAND pip3 show ${MRAY_CHECK_PIP_PACKAGE_PACKAGE_NAME}
                    RESULT_VARIABLE EXIT_CODE
                    OUTPUT_QUIET)
    if (NOT ${EXIT_CODE} EQUAL 0)
        message(FATAL_ERROR
            "The \"${MRAY_CHECK_PIP_PACKAGE_PACKAGE_NAME}\"
            Python3 package is not installed!")
    endif()

endmacro(mray_check_pip_package)

# Find the Sphinx etc..
find_package(Python3 REQUIRED COMPONENTS Interpreter)
mray_check_pip_package(PACKAGE_NAME sphinx)
mray_check_pip_package(PACKAGE_NAME furo)
mray_check_pip_package(PACKAGE_NAME myst-parser)

set(MRAY_DOCS_OUT_DIRECTORY ${MRAY_BIN_DIRECTORY}/Docs)
set(MRAY_DOCS_FILTER_FOLDER Docs)
set(MRAY_DOCS_BUILD_COMMAND ${Python3_EXECUTABLE} -m sphinx
    -D release="v${MRAY_VERSION_MAJOR}.${MRAY_VERSION_MINOR}.${MRAY_VERSION_PATCH}"
    -D project=${MRAY_PROJECT_NAME})


# Get the sources to IDE (Visual Studio)
set(MRAY_DOCS_SRC
    ${MRAY_DOCS_DIRECTORY}/conf.py
    ${MRAY_DOCS_DIRECTORY}/index.md

    ${MRAY_DOCS_DIRECTORY}/scene/scene.md

    # Statics
    ${MRAY_DOCS_DIRECTORY}/_static/cBox.frag
    ${MRAY_DOCS_DIRECTORY}/_static/cBox.js
    ${MRAY_DOCS_DIRECTORY}/_static/custom.css

    # Templates
    ${MRAY_DOCS_DIRECTORY}/_templates/path_trace.html)

source_group(TREE ${MRAY_DOCS_DIRECTORY} FILES ${MRAY_DOCS_SRC})

# check if we are building with visual studio
if(MSVC)
    set(MRAY_DOCS_BUILD_COMMAND ${MRAY_DOCS_BUILD_COMMAND} --no-color)
endif()

# Generate Build Targets
add_custom_target(MRayDocs_HTML
    COMMAND ${MRAY_DOCS_BUILD_COMMAND} -b html . ${MRAY_DOCS_OUT_DIRECTORY}/HTML
    SOURCES ${MRAY_DOCS_SRC}
    WORKING_DIRECTORY ${MRAY_DOCS_DIRECTORY}
    COMMENT "Generating MRay Documentation (HTML)")
set_target_properties(MRayDocs_HTML PROPERTIES FOLDER ${MRAY_DOCS_FILTER_FOLDER})
set_target_properties(MRayDocs_HTML PROPERTIES EXCLUDE_FROM_ALL TRUE)
set_target_properties(MRayDocs_HTML PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD TRUE)

# TODO: Add latex later
# add_custom_target(MRayDocs_PDF
#                   COMMAND ${MRAY_DOCS_BUILD_COMMAND} -b latex . ${MRAY_DOCS_OUT_DIRECTORY}/LateX
#                   SOURCES ${MRAY_DOCS_SRC}
#                   WORKING_DIRECTORY ${MRAY_DOCS_DIRECTORY}
#                   COMMENT "Generating MRay Documentation (PDF)")
# set_target_properties(MRayDocs_PDF PROPERTIES FOLDER ${MRAY_DOCS_FILTER_FOLDER})
# set_target_properties(MRayDocs_PDF PROPERTIES EXCLUDE_FROM_ALL TRUE)
# set_target_properties(MRayDocs_PDF PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD TRUE)