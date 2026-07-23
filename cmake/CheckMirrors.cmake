if(NOT DEFINED PICASSO_SOURCE_DIR)
  message(FATAL_ERROR "PICASSO_SOURCE_DIR must point to the repository root")
endif()

include("${PICASSO_SOURCE_DIR}/cmake/PicassoSources.cmake")
set(PICASSO_MIRROR_FAILURES "")

# Discover both native trees independently.  This is deliberately separate
# from the CMake configure-time inventory check: the mirror gate must report
# files that exist only in the R package as well as root-only files.
file(GLOB_RECURSE root_native_sources
     RELATIVE "${PICASSO_SOURCE_DIR}"
     "${PICASSO_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM root_native_sources "src/cli_main.cpp")
list(SORT root_native_sources)

file(GLOB_RECURSE r_native_relative
     RELATIVE "${PICASSO_SOURCE_DIR}/R-package/src"
     "${PICASSO_SOURCE_DIR}/R-package/src/*.cpp")
list(REMOVE_ITEM r_native_relative "picasso-all0.cpp" "picasso_R.cpp")
set(r_native_sources "")
foreach(source IN LISTS r_native_relative)
  # The bundled Eigen headers are vendored dependencies, not PICASSO mirrors.
  if(NOT source MATCHES "^include/eigen3/")
    list(APPEND r_native_sources "src/${source}")
  endif()
endforeach()
list(SORT r_native_sources)

foreach(source IN LISTS root_native_sources)
  list(FIND PICASSO_SOURCES "${source}" declared_index)
  if(declared_index EQUAL -1)
    list(APPEND PICASSO_MIRROR_FAILURES
         "root production source missing from PicassoSources.cmake: ${source}")
  endif()
  list(FIND r_native_sources "${source}" r_source_index)
  if(r_source_index EQUAL -1)
    list(APPEND PICASSO_MIRROR_FAILURES
         "missing R native-source mirror: ${source}")
  endif()
endforeach()
foreach(source IN LISTS PICASSO_SOURCES)
  list(FIND root_native_sources "${source}" root_source_index)
  if(root_source_index EQUAL -1)
    list(APPEND PICASSO_MIRROR_FAILURES
         "declared production source missing from root tree: ${source}")
  endif()
endforeach()
foreach(source IN LISTS r_native_sources)
  list(FIND root_native_sources "${source}" root_source_index)
  if(root_source_index EQUAL -1)
    list(APPEND PICASSO_MIRROR_FAILURES
         "R-only native source without root mirror: ${source}")
  endif()
endforeach()

# Production pairs follow the authoritative inventory.  Public headers are
# recursively discovered on both sides, so nested additions cannot escape the
# mirror gate.
set(PICASSO_MIRROR_PAIRS "")
foreach(source IN LISTS PICASSO_SOURCES)
  list(APPEND PICASSO_MIRROR_PAIRS
       "${source}|R-package/${source}")
endforeach()

set(root_public_header_dir "${PICASSO_SOURCE_DIR}/include/picasso")
set(r_public_header_dir
    "${PICASSO_SOURCE_DIR}/R-package/src/include/picasso")
if(NOT IS_DIRECTORY "${root_public_header_dir}")
  list(APPEND PICASSO_MIRROR_FAILURES
       "missing root public-header directory: include/picasso")
endif()
if(NOT IS_DIRECTORY "${r_public_header_dir}")
  list(APPEND PICASSO_MIRROR_FAILURES
       "missing R public-header directory: R-package/src/include/picasso")
endif()

file(GLOB_RECURSE root_public_headers
     RELATIVE "${root_public_header_dir}"
     "${root_public_header_dir}/*.h"
     "${root_public_header_dir}/*.hpp")
file(GLOB_RECURSE r_public_headers
     RELATIVE "${r_public_header_dir}"
     "${r_public_header_dir}/*.h"
     "${r_public_header_dir}/*.hpp")
list(SORT root_public_headers)
list(SORT r_public_headers)
if(NOT root_public_headers)
  list(APPEND PICASSO_MIRROR_FAILURES
       "no root public headers found under include/picasso")
endif()

foreach(header IN LISTS root_public_headers)
  list(FIND r_public_headers "${header}" r_header_index)
  if(r_header_index EQUAL -1)
    list(APPEND PICASSO_MIRROR_FAILURES
         "missing R public-header mirror: ${header}")
  endif()
  list(APPEND PICASSO_MIRROR_PAIRS
       "include/picasso/${header}|R-package/src/include/picasso/${header}")
endforeach()
foreach(header IN LISTS r_public_headers)
  list(FIND root_public_headers "${header}" root_header_index)
  if(root_header_index EQUAL -1)
    list(APPEND PICASSO_MIRROR_FAILURES
         "R-only public header without root mirror: ${header}")
  endif()
endforeach()

# Private native headers are not installed, but they are compiled independently
# by the root and R-package builds and therefore need the same two-way drift
# protection as production sources and public headers.
set(root_private_header_dir "${PICASSO_SOURCE_DIR}/src/internal")
set(r_private_header_dir "${PICASSO_SOURCE_DIR}/R-package/src/internal")
if(NOT IS_DIRECTORY "${root_private_header_dir}")
  list(APPEND PICASSO_MIRROR_FAILURES
       "missing root private-header directory: src/internal")
endif()
if(NOT IS_DIRECTORY "${r_private_header_dir}")
  list(APPEND PICASSO_MIRROR_FAILURES
       "missing R private-header directory: R-package/src/internal")
endif()

file(GLOB_RECURSE root_private_headers
     RELATIVE "${root_private_header_dir}"
     "${root_private_header_dir}/*.h"
     "${root_private_header_dir}/*.hpp")
file(GLOB_RECURSE r_private_headers
     RELATIVE "${r_private_header_dir}"
     "${r_private_header_dir}/*.h"
     "${r_private_header_dir}/*.hpp")
list(SORT root_private_headers)
list(SORT r_private_headers)
if(NOT root_private_headers)
  list(APPEND PICASSO_MIRROR_FAILURES
       "no root private headers found under src/internal")
endif()

foreach(header IN LISTS root_private_headers)
  list(FIND r_private_headers "${header}" r_header_index)
  if(r_header_index EQUAL -1)
    list(APPEND PICASSO_MIRROR_FAILURES
         "missing R private-header mirror: ${header}")
  endif()
  list(APPEND PICASSO_MIRROR_PAIRS
       "src/internal/${header}|R-package/src/internal/${header}")
endforeach()
foreach(header IN LISTS r_private_headers)
  list(FIND root_private_headers "${header}" root_header_index)
  if(root_header_index EQUAL -1)
    list(APPEND PICASSO_MIRROR_FAILURES
         "R-only private header without root mirror: ${header}")
  endif()
endforeach()

# Unity builds are an ordered representation of the same production set.
# Compare their raw normalized order before sorting; otherwise a one-sided
# reorder is invisible and the two package builds can drift subtly.
set(root_unity "${PICASSO_SOURCE_DIR}/amalgamation/picasso-all0.cpp")
set(r_unity "${PICASSO_SOURCE_DIR}/R-package/src/picasso-all0.cpp")
set(root_unity_sources "")
set(r_unity_sources "")

if(EXISTS "${root_unity}")
  file(STRINGS "${root_unity}" root_unity_lines
       REGEX "^[ \t]*#include[ \t]+.*\\.cpp")
  foreach(line IN LISTS root_unity_lines)
    if(line MATCHES "^[ \t]*#include[ \t]+\"\\.\\./(src/[^\"]+\\.cpp)\"[ \t]*$")
      list(APPEND root_unity_sources "${CMAKE_MATCH_1}")
    else()
      list(APPEND PICASSO_MIRROR_FAILURES
           "unrecognized C++ include in amalgamation/picasso-all0.cpp: ${line}")
    endif()
  endforeach()
else()
  list(APPEND PICASSO_MIRROR_FAILURES
       "missing root unity build: amalgamation/picasso-all0.cpp")
endif()

if(EXISTS "${r_unity}")
  file(STRINGS "${r_unity}" r_unity_lines
       REGEX "^[ \t]*#include[ \t]+.*\\.cpp")
  foreach(line IN LISTS r_unity_lines)
    if(line MATCHES "^[ \t]*#include[ \t]+\"([^\"]+\\.cpp)\"[ \t]*$")
      list(APPEND r_unity_sources "src/${CMAKE_MATCH_1}")
    else()
      list(APPEND PICASSO_MIRROR_FAILURES
           "unrecognized C++ include in R-package/src/picasso-all0.cpp: ${line}")
    endif()
  endforeach()
else()
  list(APPEND PICASSO_MIRROR_FAILURES
       "missing R unity build: R-package/src/picasso-all0.cpp")
endif()

if(NOT "${root_unity_sources}" STREQUAL "${r_unity_sources}")
  list(APPEND PICASSO_MIRROR_FAILURES
       "root and R unity builds have different raw include order")
endif()

set(root_unity_sorted ${root_unity_sources})
set(r_unity_sorted ${r_unity_sources})
set(expected_unity_sorted ${PICASSO_SOURCES})
list(SORT root_unity_sorted)
list(SORT r_unity_sorted)
list(SORT expected_unity_sorted)
if(NOT "${root_unity_sorted}" STREQUAL "${expected_unity_sorted}")
  list(APPEND PICASSO_MIRROR_FAILURES
       "amalgamation/picasso-all0.cpp does not exactly match the production source inventory")
endif()
if(NOT "${r_unity_sorted}" STREQUAL "${expected_unity_sorted}")
  list(APPEND PICASSO_MIRROR_FAILURES
       "R-package/src/picasso-all0.cpp does not exactly match the production source inventory")
endif()

foreach(pair IN LISTS PICASSO_MIRROR_PAIRS)
  string(REPLACE "|" ";" paths "${pair}")
  list(GET paths 0 root_relative)
  list(GET paths 1 r_relative)
  set(root_file "${PICASSO_SOURCE_DIR}/${root_relative}")
  set(r_file "${PICASSO_SOURCE_DIR}/${r_relative}")

  if(NOT EXISTS "${root_file}" OR NOT EXISTS "${r_file}")
    list(APPEND PICASSO_MIRROR_FAILURES
         "missing mirror pair: ${root_relative} <-> ${r_relative}")
    continue()
  endif()

  execute_process(
      COMMAND "${CMAKE_COMMAND}" -E compare_files "${root_file}" "${r_file}"
      RESULT_VARIABLE compare_result)
  if(NOT compare_result EQUAL 0)
    list(APPEND PICASSO_MIRROR_FAILURES
         "different mirror pair: ${root_relative} <-> ${r_relative}")
  endif()
endforeach()

if(PICASSO_MIRROR_FAILURES)
  string(REPLACE ";" "\n  " failure_text "${PICASSO_MIRROR_FAILURES}")
  message(FATAL_ERROR "Source mirror check failed:\n  ${failure_text}")
endif()

list(LENGTH PICASSO_MIRROR_PAIRS mirror_count)
message(STATUS "Source mirror check passed (${mirror_count} pairs)")
