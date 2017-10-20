cmake_minimum_required(VERSION 3.9)
project(picasso)
include(cmake/Utils.cmake)
find_package(OpenMP)

set_default_configuration_release()
msvc_use_static_runtime()

# Compiler flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BUILD_USE_64BITS on)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
endif (CMAKE_SIZEOF_VOID_P EQUAL 8 )
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    # Multithreaded compilation
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
    # Correct error for GCC 5 and cuda
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
    # Performance
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
endif()


include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/dmlc-core/include
        )

file(GLOB_RECURSE SOURCES
        src/*.cpp
        src/*.cc
        src/*.h
        include/*.h
        include/*.hpp
        )
# Only add main function for executable target (---no cli version yet---)
#list(REMOVE_ITEM SOURCES ${PROJECT_SOURCE_DIR}/src/cli_main.cpp)

# dmlc-core
#add_subdirectory(dmlc-core)
#set(LINK_LIBRARIES dmlccore)

add_library(picassoobj OBJECT ${SOURCES})

# Executable (----no cli version yet----)
#add_executable(clipicasso $<TARGET_OBJECTS:objpicasso> src/cli_main.cc)
#set_target_properties(clipicasso PROPERTIES
#     OUTPUT_NAME picasso
#     )
#set_output_directory(clipicasso ${PROJECT_SOURCE_DIR})
#target_link_libraries(clipicasso ${LINK_LIBRARIES})

add_library(picasso SHARED $<TARGET_OBJECTS:picassoobj>)
target_link_libraries(picasso ${LINK_LIBRARIES})
set_output_directory(picasso ${PROJECT_SOURCE_DIR}/lib)
if(MINGW)
    # remove the 'lib' prefix to conform to windows convention for shared library names
    set_target_properties(picasso PROPERTIES PREFIX "")
endif()

#Ensure these two targets do not build simultaneously, as they produce outputs with conflicting names (---no cli version yet---)
#add_dependencies(picasso clipicasso)

# Group sources
auto_source_group("${SOURCES}")
