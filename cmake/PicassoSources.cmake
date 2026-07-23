# Authoritative production-source inventory shared by the build and mirror
# checks.  Keep the scalar subset separate because the Gaussian C API test
# links it directly to exercise allocation-failure paths.  The optional
# command-line entry point is not part of this library.
set(PICASSO_SCALAR_SOURCES
    src/c_api/c_api.cpp
    src/objective/gaussian_cov_update.cpp
    src/objective/gaussian_naive_update.cpp
    src/objective/glm.cpp
    src/objective/sqrtmse.cpp
    src/solver/actgd.cpp
    src/solver/actnewton.cpp
    src/solver/solver_params.cpp)

set(PICASSO_MULTINOMIAL_SOURCES
    src/c_api/multinomial.cpp
    src/objective/multinomial_objective.cpp
    src/solver/multinomial_actnewton.cpp
    src/solver/multinomial_lla.cpp)

set(PICASSO_SOURCES
    ${PICASSO_SCALAR_SOURCES}
    ${PICASSO_MULTINOMIAL_SOURCES})

function(picasso_validate_source_inventory source_dir)
  file(GLOB_RECURSE discovered_sources
       RELATIVE "${source_dir}"
       "${source_dir}/src/*.cpp")
  list(REMOVE_ITEM discovered_sources "src/cli_main.cpp")
  list(SORT discovered_sources)

  set(declared_sources ${PICASSO_SOURCES})
  list(SORT declared_sources)
  if(NOT "${discovered_sources}" STREQUAL "${declared_sources}")
    string(REPLACE ";" "\n  " discovered_text "${discovered_sources}")
    string(REPLACE ";" "\n  " declared_text "${declared_sources}")
    message(FATAL_ERROR
            "Production source inventory differs from PicassoSources.cmake.\n"
            "Discovered:\n  ${discovered_text}\n"
            "Declared:\n  ${declared_text}")
  endif()
endfunction()
