if(NOT DEFINED PICASSO_SOURCE_DIR)
  message(FATAL_ERROR "PICASSO_SOURCE_DIR must point to the repository root")
endif()

set(eigen_module_dir
    "${PICASSO_SOURCE_DIR}/R-package/src/include/eigen3/Eigen")
if(NOT IS_DIRECTORY "${eigen_module_dir}")
  message(FATAL_ERROR "Bundled Eigen module directory is missing")
endif()

# Top-level files under Eigen/ are public/module headers. Internal headers under
# Eigen/src/ may intentionally share an enclosing module's warning scope, so
# they are not independent pairing units.
file(GLOB eigen_module_entries "${eigen_module_dir}/*")
set(pair_failures "")
set(paired_header_count 0)
foreach(header_path IN LISTS eigen_module_entries)
  if(IS_DIRECTORY "${header_path}")
    continue()
  endif()

  file(STRINGS "${header_path}" warning_markers REGEX
       [=[^[ 	]*#[ 	]*include[ 	]+["<]src/Core/util/(Disable|Reenable)StupidWarnings[.]h[">]]=])
  list(LENGTH warning_markers marker_count)
  if(marker_count EQUAL 0)
    continue()
  endif()

  get_filename_component(header_name "${header_path}" NAME)
  if(NOT marker_count EQUAL 2)
    list(APPEND pair_failures
         "${header_name}: expected one Disable and one Reenable include, found ${marker_count}")
    continue()
  endif()

  list(GET warning_markers 0 first_marker)
  list(GET warning_markers 1 second_marker)
  if(NOT first_marker MATCHES "DisableStupidWarnings[.]h" OR
     NOT second_marker MATCHES "ReenableStupidWarnings[.]h")
    list(APPEND pair_failures
         "${header_name}: warning includes are not ordered Disable then Reenable")
    continue()
  endif()
  math(EXPR paired_header_count "${paired_header_count} + 1")
endforeach()

if(paired_header_count EQUAL 0)
  list(APPEND pair_failures
       "no bundled Eigen public/module header contains a warning pair")
endif()
if(pair_failures)
  string(REPLACE ";" "\n  " failure_text "${pair_failures}")
  message(FATAL_ERROR
          "Bundled Eigen warning-pair check failed:\n  ${failure_text}")
endif()

message(STATUS
        "Bundled Eigen warning-pair check passed (${paired_header_count} headers)")
