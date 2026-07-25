cmake_minimum_required(VERSION 3.10)
project(picasso LANGUAGES CXX)

include(cmake/Utils.cmake)
include(cmake/PicassoSources.cmake)
include(CTest)

set_default_configuration_release()
msvc_use_static_runtime()

option(PICASSO_ENABLE_SANITIZERS
       "Build the C++ library and tests with AddressSanitizer and UBSan" OFF)

picasso_validate_source_inventory("${PROJECT_SOURCE_DIR}")

# Prefer a system Eigen package when one is configured.  The repository ships
# the complete headers used by the R package, so a clean checkout remains
# self-contained on machines without Eigen installed system-wide.
find_package(Eigen3 3.2 QUIET NO_MODULE)
if(TARGET Eigen3::Eigen)
  set(PICASSO_EIGEN_TARGET Eigen3::Eigen)
  message(STATUS "Using configured Eigen3 package")
else()
  set(PICASSO_BUNDLED_EIGEN_DIR
      "${PROJECT_SOURCE_DIR}/R-package/src/include/eigen3")
  if(NOT EXISTS "${PICASSO_BUNDLED_EIGEN_DIR}/Eigen/Dense")
    message(FATAL_ERROR
            "Eigen3 was not found and the bundled R-package Eigen headers "
            "are missing")
  endif()
  add_library(picasso_eigen INTERFACE)
  target_include_directories(picasso_eigen SYSTEM INTERFACE
      "${PICASSO_BUNDLED_EIGEN_DIR}")
  set(PICASSO_EIGEN_TARGET picasso_eigen)
  message(STATUS "Using bundled Eigen3: ${PICASSO_BUNDLED_EIGEN_DIR}")
endif()

find_package(OpenMP QUIET)

function(picasso_enable_sanitizers target)
  if(NOT PICASSO_ENABLE_SANITIZERS)
    return()
  endif()
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR
            "PICASSO_ENABLE_SANITIZERS requires GCC or a Clang-family compiler")
  endif()

  target_compile_options(${target} PRIVATE
      -fsanitize=address,undefined
      -fno-omit-frame-pointer)
  # LINK_OPTIONS was introduced after the minimum supported CMake version.
  # LINK_FLAGS keeps the sanitizer runtime on the final link line with 3.10.
  set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS
               " -fsanitize=address,undefined")
endfunction()

add_library(picasso SHARED ${PICASSO_SOURCES})
target_compile_features(picasso PUBLIC cxx_std_11)
target_compile_definitions(picasso
    PRIVATE PICASSO_BUILDING_SHARED
    INTERFACE PICASSO_USING_SHARED)
target_include_directories(picasso PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(picasso PUBLIC ${PICASSO_EIGEN_TARGET})
set_target_properties(picasso PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)
if(OpenMP_CXX_FOUND)
  target_link_libraries(picasso PRIVATE OpenMP::OpenMP_CXX)
endif()
if(MSVC)
  target_compile_options(picasso PRIVATE /MP)
else()
  target_compile_options(picasso PRIVATE
      $<$<CONFIG:Release>:-funroll-loops>)
endif()
picasso_enable_sanitizers(picasso)

# Keep each configure/build tree self-contained.  CMake's default output
# layout also keeps configurations separate for multi-config generators, so a
# sanitizer build can never overwrite a Release library from another tree.
if(MINGW)
  set_target_properties(picasso PROPERTIES PREFIX "")
endif()

# Copy a deliberately selected build into a packaging/staging directory only
# when this explicit target is requested.  The destination remains
# configurable so callers can stage directly into an isolated Python package
# tree instead of sharing the repository's historical lib/ directory.
set(PICASSO_STAGE_DIRECTORY "${CMAKE_BINARY_DIR}/stage" CACHE PATH
    "Destination used by the explicit stage_picasso target")
add_custom_target(stage_picasso
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PICASSO_STAGE_DIRECTORY}"
    COMMAND ${CMAKE_COMMAND} -E remove -f
            "${PICASSO_STAGE_DIRECTORY}/libpicasso.so"
            "${PICASSO_STAGE_DIRECTORY}/libpicasso.dylib"
            "${PICASSO_STAGE_DIRECTORY}/picasso.dll"
            "${PICASSO_STAGE_DIRECTORY}/libpicasso.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:picasso>"
            "${PICASSO_STAGE_DIRECTORY}/$<TARGET_FILE_NAME:picasso>"
    DEPENDS picasso
    COMMENT "Staging picasso native library in ${PICASSO_STAGE_DIRECTORY}"
    VERBATIM)

set(PICASSO_MIRROR_CHECK_COMMAND
    ${CMAKE_COMMAND}
    -DPICASSO_SOURCE_DIR=${PROJECT_SOURCE_DIR}
    -P ${PROJECT_SOURCE_DIR}/cmake/CheckMirrors.cmake)
add_custom_target(check_mirrors
    COMMAND ${PICASSO_MIRROR_CHECK_COMMAND}
    COMMENT "Checking root and R-package source mirrors"
    VERBATIM)

set(PICASSO_EIGEN_WARNING_CHECK_COMMAND
    ${CMAKE_COMMAND}
    -DPICASSO_SOURCE_DIR=${PROJECT_SOURCE_DIR}
    -P ${PROJECT_SOURCE_DIR}/cmake/CheckEigenWarningPairs.cmake)
add_custom_target(check_eigen_warning_pairs
    COMMAND ${PICASSO_EIGEN_WARNING_CHECK_COMMAND}
    COMMENT "Checking bundled Eigen warning push/pop pairs"
    VERBATIM)

if(BUILD_TESTING)
  # The native tests exercise internal C++ classes that the shared library
  # deliberately does not export (on Windows only PICASSO_C_API symbols leave
  # the DLL), so link them against a static build of the same sources.
  add_library(picasso_testing STATIC ${PICASSO_SOURCES})
  target_compile_features(picasso_testing PUBLIC cxx_std_11)
  set_target_properties(picasso_testing PROPERTIES CXX_EXTENSIONS OFF)
  target_include_directories(picasso_testing PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
  target_link_libraries(picasso_testing PUBLIC ${PICASSO_EIGEN_TARGET})
  if(OpenMP_CXX_FOUND)
    target_link_libraries(picasso_testing PRIVATE OpenMP::OpenMP_CXX)
  endif()
  picasso_enable_sanitizers(picasso_testing)

  function(picasso_add_cpp_test target source)
    add_executable(${target} ${source})
    target_compile_features(${target} PRIVATE cxx_std_11)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    target_link_libraries(${target} PRIVATE picasso_testing)
    picasso_enable_sanitizers(${target})
    add_test(NAME ${target} COMMAND ${target})
  endfunction()

  picasso_add_cpp_test(multinomial_objective_test
                       tests/multinomial_objective_test.cpp)
  picasso_add_cpp_test(multinomial_actnewton_test
                       tests/multinomial_actnewton_test.cpp)
  picasso_add_cpp_test(multinomial_lla_test
                       tests/multinomial_lla_test.cpp)
  picasso_add_cpp_test(multinomial_c_api_test
                       tests/multinomial_c_api_test.cpp)
  picasso_add_cpp_test(actnewton_lla_test
                       tests/actnewton_lla_test.cpp)
  picasso_add_cpp_test(scalar_lla_c_api_test
                       tests/scalar_lla_c_api_test.cpp)
  picasso_add_cpp_test(gaussian_objective_test
                       tests/gaussian_objective_test.cpp)
  picasso_add_cpp_test(gaussian_covariance_cache_test
                       tests/gaussian_covariance_cache_test.cpp)
  # Compile the scalar C ABI into this test executable so its controlled
  # operator-new failure also reaches implementation allocations on macOS.
  add_executable(gaussian_c_api_test
      tests/gaussian_c_api_test.cpp
      ${PICASSO_SCALAR_SOURCES})
  target_compile_features(gaussian_c_api_test PRIVATE cxx_std_11)
  target_include_directories(gaussian_c_api_test PRIVATE
      "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(gaussian_c_api_test PRIVATE ${PICASSO_EIGEN_TARGET})
  set_target_properties(gaussian_c_api_test PROPERTIES CXX_EXTENSIONS OFF)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(gaussian_c_api_test PRIVATE OpenMP::OpenMP_CXX)
  endif()
  picasso_enable_sanitizers(gaussian_c_api_test)
  add_test(NAME gaussian_c_api_test COMMAND gaussian_c_api_test)
  picasso_add_cpp_test(gaussian_iteration_status_test
                       tests/gaussian_iteration_status_test.cpp)
  picasso_add_cpp_test(actgd_compact_path_test
                       tests/actgd_compact_path_test.cpp)
  picasso_add_cpp_test(glm_numerical_stability_test
                       tests/glm_numerical_stability_test.cpp)
  picasso_add_cpp_test(glm_offset_test
                       tests/glm_offset_test.cpp)
  picasso_add_cpp_test(sqrt_constant_response_test
                       tests/sqrt_constant_response_test.cpp)
  picasso_add_cpp_test(scalar_borrowed_design_test
                       tests/scalar_borrowed_design_test.cpp)
  # The previous standalone implementation is a test oracle only. Keep it
  # out of every production library and export table.
  target_sources(multinomial_c_api_test PRIVATE
                 tests/multinomial_legacy_oracle.cpp)

  add_test(NAME source_mirrors COMMAND ${PICASSO_MIRROR_CHECK_COMMAND})
  add_test(NAME source_mirror_negative_probes
           COMMAND ${CMAKE_COMMAND}
                   -DPICASSO_SOURCE_DIR=${PROJECT_SOURCE_DIR}
                   -DPICASSO_PROBE_DIR=${CMAKE_BINARY_DIR}/mirror-negative-probes
                   -P ${PROJECT_SOURCE_DIR}/cmake/CheckMirrorNegativeProbes.cmake)
  add_test(NAME eigen_warning_pairs
           COMMAND ${PICASSO_EIGEN_WARNING_CHECK_COMMAND})
endif()

auto_source_group("${PICASSO_SOURCES}")
