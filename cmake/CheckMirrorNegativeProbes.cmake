if(NOT DEFINED PICASSO_SOURCE_DIR OR NOT DEFINED PICASSO_PROBE_DIR)
  message(FATAL_ERROR
          "PICASSO_SOURCE_DIR and PICASSO_PROBE_DIR must both be defined")
endif()

set(mirror_check "${PICASSO_SOURCE_DIR}/cmake/CheckMirrors.cmake")

function(prepare_probe_fixture name out_fixture)
  set(fixture "${PICASSO_PROBE_DIR}/${name}")
  file(REMOVE_RECURSE "${fixture}")
  file(MAKE_DIRECTORY
       "${fixture}/cmake"
       "${fixture}/include"
       "${fixture}/amalgamation"
       "${fixture}/R-package/src/include"
       "${fixture}/R-package/src/internal")
  file(COPY "${PICASSO_SOURCE_DIR}/cmake/PicassoSources.cmake"
       DESTINATION "${fixture}/cmake")
  file(COPY "${PICASSO_SOURCE_DIR}/src" DESTINATION "${fixture}")
  file(COPY "${PICASSO_SOURCE_DIR}/include/picasso"
       DESTINATION "${fixture}/include")
  foreach(directory c_api objective solver internal)
    file(COPY "${PICASSO_SOURCE_DIR}/R-package/src/${directory}"
         DESTINATION "${fixture}/R-package/src")
  endforeach()
  file(COPY "${PICASSO_SOURCE_DIR}/R-package/src/include/picasso"
       DESTINATION "${fixture}/R-package/src/include")
  file(COPY "${PICASSO_SOURCE_DIR}/amalgamation/picasso-all0.cpp"
       DESTINATION "${fixture}/amalgamation")
  file(COPY "${PICASSO_SOURCE_DIR}/R-package/src/picasso-all0.cpp"
       DESTINATION "${fixture}/R-package/src")
  set(${out_fixture} "${fixture}" PARENT_SCOPE)
endfunction()

function(run_mirror_check fixture expected_result expected_text)
  execute_process(
      COMMAND "${CMAKE_COMMAND}"
              "-DPICASSO_SOURCE_DIR=${fixture}"
              -P "${mirror_check}"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE stdout
      ERROR_VARIABLE stderr)
  set(output "${stdout}\n${stderr}")
  if(expected_result STREQUAL "PASS")
    if(NOT result EQUAL 0)
      message(FATAL_ERROR
              "Pristine mirror fixture unexpectedly failed:\n${output}")
    endif()
  else()
    if(result EQUAL 0)
      message(FATAL_ERROR
              "Negative mirror probe unexpectedly passed: ${expected_text}")
    endif()
    string(FIND "${output}" "${expected_text}" expected_index)
    if(expected_index EQUAL -1)
      message(FATAL_ERROR
              "Negative mirror probe failed for the wrong reason.\n"
              "Expected: ${expected_text}\nOutput:\n${output}")
    endif()
  endif()
endfunction()

prepare_probe_fixture(pristine fixture)
run_mirror_check("${fixture}" PASS "Source mirror check passed")

prepare_probe_fixture(root-only-cpp fixture)
file(WRITE "${fixture}/src/objective/root_only_probe.cpp" "// probe\n")
run_mirror_check("${fixture}" FAIL
                 "root production source missing from PicassoSources.cmake")

prepare_probe_fixture(r-only-cpp fixture)
file(WRITE "${fixture}/R-package/src/objective/r_only_probe.cpp" "// probe\n")
run_mirror_check("${fixture}" FAIL "R-only native source without root mirror")

prepare_probe_fixture(nested-header fixture)
file(MAKE_DIRECTORY "${fixture}/include/picasso/detail")
file(WRITE "${fixture}/include/picasso/detail/root_only_probe.hpp" "// probe\n")
run_mirror_check("${fixture}" FAIL "missing R public-header mirror")

prepare_probe_fixture(root-only-private-header fixture)
file(WRITE "${fixture}/src/internal/root_only_probe.hpp" "// probe\n")
run_mirror_check("${fixture}" FAIL "missing R private-header mirror")

prepare_probe_fixture(r-only-private-header fixture)
file(WRITE "${fixture}/R-package/src/internal/r_only_probe.hpp" "// probe\n")
run_mirror_check("${fixture}" FAIL
                 "R-only private header without root mirror")

prepare_probe_fixture(private-header-drift fixture)
file(APPEND
     "${fixture}/R-package/src/internal/multinomial_problem_view.hpp"
     "\n// drift probe\n")
run_mirror_check(
    "${fixture}" FAIL
    "different mirror pair: src/internal/multinomial_problem_view.hpp")

prepare_probe_fixture(unity-reorder fixture)
set(unity_file "${fixture}/amalgamation/picasso-all0.cpp")
file(READ "${unity_file}" unity_contents)
set(original_pair
    "#include \"../src/objective/gaussian_naive_update.cpp\"\n#include \"../src/objective/gaussian_cov_update.cpp\"")
set(swapped_pair
    "#include \"../src/objective/gaussian_cov_update.cpp\"\n#include \"../src/objective/gaussian_naive_update.cpp\"")
string(FIND "${unity_contents}" "${original_pair}" original_pair_index)
if(original_pair_index EQUAL -1)
  message(FATAL_ERROR "Unity reorder probe could not find its source include pair")
endif()
string(REPLACE "${original_pair}" "${swapped_pair}"
       unity_contents "${unity_contents}")
file(WRITE "${unity_file}" "${unity_contents}")
run_mirror_check("${fixture}" FAIL
                 "root and R unity builds have different raw include order")

prepare_probe_fixture(r-unity-reorder fixture)
set(unity_file "${fixture}/R-package/src/picasso-all0.cpp")
file(READ "${unity_file}" unity_contents)
set(original_pair
    "#include \"objective/gaussian_naive_update.cpp\"\n#include \"objective/gaussian_cov_update.cpp\"")
set(swapped_pair
    "#include \"objective/gaussian_cov_update.cpp\"\n#include \"objective/gaussian_naive_update.cpp\"")
string(FIND "${unity_contents}" "${original_pair}" original_pair_index)
if(original_pair_index EQUAL -1)
  message(FATAL_ERROR "R unity reorder probe could not find its source include pair")
endif()
string(REPLACE "${original_pair}" "${swapped_pair}"
       unity_contents "${unity_contents}")
file(WRITE "${unity_file}" "${unity_contents}")
run_mirror_check("${fixture}" FAIL
                 "root and R unity builds have different raw include order")

message(STATUS "All isolated source-mirror negative probes failed as expected")
