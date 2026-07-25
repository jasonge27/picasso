#ifndef EIGEN_WARNINGS_DISABLED
#define EIGEN_WARNINGS_DISABLED

// This vendored Eigen copy keeps warning policy under the caller's control,
// but ReenableStupidWarnings.h still restores a diagnostic state. Preserve a
// matching state here without disabling any warnings ourselves.
#ifdef _MSC_VER
  #pragma warning( push )
#elif defined __INTEL_COMPILER
  #pragma warning push
#elif defined __clang__
  #pragma clang diagnostic push
#elif defined __GNUC__ && __GNUC__>=6
  #pragma GCC diagnostic push
#endif

#endif  // not EIGEN_WARNINGS_DISABLED
