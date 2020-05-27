#include_directories(${CMAKE_SOURCE_DIR}/include algorithms ext util nn .)

message(STATUS "VTOOL_CHECKPOINT = TRUE")

message(STATUS "VTOOL_IBEIS_VERSION = ${VTOOL_IBEIS_VERSION}")
set(VTOOL_VERSION ${VTOOL_IBEIS_VERSION})
message(STATUS "VTOOL_VERSION = ${VTOOL_VERSION}")
add_definitions(-D_VTOOL_IBEIS_VERSION=${VTOOL_IBEIS_VERSION})
add_definitions(-DVTOOL_IBEIS_VERSION_=${VTOOL_IBEIS_VERSION})
add_definitions(-DVTOOL_IBEIS_VERSION=${VTOOL_IBEIS_VERSION})
add_definitions(-D_VTOOL_VERSION=${VTOOL_VERSION})
add_definitions(-DVTOOL_VERSION_=${VTOOL_VERSION})
add_definitions(-DVTOOL_VERSION=${VTOOL_VERSION})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vtool/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/vtool/config.h)

file(GLOB_RECURSE CPP_SOURCES sver.cpp)

add_library(vtool_cpp_s STATIC ${CPP_SOURCES})
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANG)
    set_target_properties(vtool_cpp_s PROPERTIES COMPILE_FLAGS -fPIC)
endif()
set_property(TARGET vtool_cpp_s PROPERTY COMPILE_DEFINITIONS VTOOL_STATIC)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_COMPILER_IS_GNUCC)
    set_target_properties(vtool_cpp PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(vtool_cpp -Wl,-whole-archive vtool_cpp_s -Wl,-no-whole-archive)
else()
    add_library(vtool_cpp SHARED ${CPP_SOURCES})
    set_target_properties(vtool_cpp PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS YES)
endif()

set_target_properties(vtool_cpp PROPERTIES
   VERSION ${VTOOL_IBEIS_VERSION}
   SOVERSION ${VTOOL_SOVERSION}
   DEFINE_SYMBOL VTOOL_EXPORTS
)

if(NOT SKBUILD)
    # Skip this part if running via setup.py

    install (
        TARGETS vtool_cpp vtool_cpp_s
        EXPORT ${targets_export_name}
        INCLUDES DESTINATION include
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION ${VTOOL_LIB_INSTALL_DIR}
        ARCHIVE DESTINATION ${VTOOL_LIB_INSTALL_DIR}
    )

    install (
        DIRECTORY vtool
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )

else()
    # Only install the library files when building with skbuild
    install (
        TARGETS vtool_cpp vtool_cpp_s
        EXPORT ${targets_export_name}
        LIBRARY DESTINATION ${VTOOL_LIB_INSTALL_DIR}
        ARCHIVE DESTINATION ${VTOOL_LIB_INSTALL_DIR}
    )
endif()
