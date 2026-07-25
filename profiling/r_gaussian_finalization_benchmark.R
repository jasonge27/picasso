#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L) {
  stop(paste0(
    "usage: Rscript profiling/r_gaussian_finalization_benchmark.R ",
    "OLD_SOURCE NEW_SOURCE PICASSO_LIBRARY OUTPUT_CSV [REPETITIONS]"
  ), call. = FALSE)
}

old.source <- normalizePath(args[[1L]], mustWork = TRUE)
new.source <- normalizePath(args[[2L]], mustWork = TRUE)
picasso.library <- normalizePath(args[[3L]], mustWork = TRUE)
output.csv <- args[[4L]]
repetitions <- if (length(args) >= 5L) as.integer(args[[5L]]) else 7L
if (is.na(repetitions) || repetitions < 7L) {
  stop("REPETITIONS must be an integer greater than or equal to 7.",
       call. = FALSE)
}

.libPaths(c(picasso.library, .libPaths()))
suppressPackageStartupMessages(library(picasso))

namespace <- asNamespace("picasso")
old.environment <- new.env(parent = namespace)
new.environment <- new.env(parent = namespace)
sys.source(old.source, old.environment)
sys.source(new.source, new.environment)

old.fit <- old.environment$picasso.gaussian
new.fit <- new.environment$picasso.gaussian

source.hash <- c(
  old = unname(tools::md5sum(old.source)),
  new = unname(tools::md5sum(new.source))
)
native.files <- list.files(
  system.file("libs", package = "picasso"),
  pattern = "\\.(so|dylib|dll)$", full.names = TRUE
)
if (length(native.files) != 1L) {
  stop("Expected exactly one installed picasso native library.", call. = FALSE)
}
native.hash <- unname(tools::md5sum(native.files[[1L]]))

detect.cpu <- function() {
  explicit <- Sys.getenv("PICASSO_BENCHMARK_CPU", unset = "")
  if (nzchar(explicit)) {
    return(explicit)
  }

  system.info <- Sys.info()
  if (identical(unname(system.info[["sysname"]]), "Darwin")) {
    sysctl <- Sys.which("sysctl")
    if (nzchar(sysctl)) {
      for (key in c("machdep.cpu.brand_string", "hw.model")) {
        value <- suppressWarnings(tryCatch(
          system2(sysctl, c("-n", key), stdout = TRUE, stderr = TRUE),
          error = function(condition) character()
        ))
        if (length(value) > 0L &&
            (is.null(attr(value, "status")) || attr(value, "status") == 0L)) {
          value <- trimws(paste(value, collapse = " "))
          if (nzchar(value)) {
            return(value)
          }
        }
      }
    }

    system.profiler <- "/usr/sbin/system_profiler"
    if (file.exists(system.profiler)) {
      hardware <- suppressWarnings(tryCatch(
        system2(
          system.profiler,
          c("SPHardwareDataType", "-detailLevel", "mini"),
          stdout = TRUE, stderr = FALSE
        ),
        error = function(condition) character()
      ))
      chip <- grep(
        "^[[:space:]]*(Chip|Processor Name):", hardware, value = TRUE
      )
      if (length(chip) > 0L) {
        return(trimws(sub("^[^:]*:", "", chip[[1L]])))
      }
    }
  }

  if (identical(unname(system.info[["sysname"]]), "Linux") &&
      file.exists("/proc/cpuinfo")) {
    cpu.info <- readLines("/proc/cpuinfo", warn = FALSE)
    model <- grep("^model name[[:space:]]*:", cpu.info, value = TRUE)
    if (length(model) > 0L) {
      return(trimws(sub("^[^:]*:", "", model[[1L]])))
    }
  }

  windows.cpu <- Sys.getenv("PROCESSOR_IDENTIFIER", unset = "")
  if (nzchar(windows.cpu)) {
    return(windows.cpu)
  }
  unname(system.info[["machine"]])
}

thread.variables <- c(
  "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
  "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "RCPP_PARALLEL_NUM_THREADS"
)
thread.values <- Sys.getenv(thread.variables, unset = "<unset>")
names(thread.values) <- thread.variables
software.versions <- extSoftVersion()
system.info <- Sys.info()
script.argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script.path <- if (length(script.argument) == 1L) {
  normalizePath(sub("^--file=", "", script.argument), mustWork = TRUE)
} else {
  NA_character_
}
provenance <- list(
  r.version = R.version.string,
  matrix.version = as.character(utils::packageVersion("Matrix")),
  blas = if ("BLAS" %in% names(software.versions)) {
    unname(software.versions[["BLAS"]])
  } else {
    "<unknown>"
  },
  cpu = detect.cpu(),
  platform = paste(
    R.version$platform, system.info[["sysname"]], system.info[["release"]],
    system.info[["machine"]], sep = " | "
  ),
  threads = paste(
    paste0(names(thread.values), "=", unname(thread.values)), collapse = ";"
  ),
  native.path = normalizePath(native.files[[1L]], mustWork = TRUE),
  script.hash = if (is.na(script.path)) {
    NA_character_
  } else {
    unname(tools::md5sum(script.path))
  }
)

object.checksum <- function(object) {
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(object, path, version = 3L, compress = FALSE)
  unname(tools::md5sum(path))
}

normalize.fit <- function(fit) {
  fit$runtime <- structure(0, units = "secs", class = "difftime")
  fit
}

profile.once <- function(operation, inner = 1L) {
  memory.path <- tempfile()
  on.exit(unlink(memory.path), add = TRUE)
  gc()
  Rprofmem(memory.path)
  timing <- tryCatch(
    system.time({
      for (iteration in seq_len(inner)) result <- operation()
    })[["elapsed"]],
    finally = Rprofmem(NULL)
  )
  records <- readLines(memory.path, warn = FALSE)
  sizes <- suppressWarnings(as.numeric(sub(" .*", "", records)))
  list(
    result = result,
    inner = inner,
    batch.elapsed.seconds = as.numeric(timing),
    elapsed.seconds = as.numeric(timing) / inner,
    allocated.bytes = sum(sizes, na.rm = TRUE) / inner
  )
}

calibrate.inner <- function(operations, initial, target.seconds = 0.2) {
  required <- vapply(operations, function(operation) {
    inner <- initial
    repeat {
      timing <- system.time({
        for (iteration in seq_len(inner)) result <- operation()
      })[["elapsed"]]
      if (timing >= target.seconds) {
        return(inner)
      }
      multiplier <- if (timing > 0) {
        max(1.5, 1.25 * target.seconds / timing)
      } else {
        10.0
      }
      inner <- as.integer(ceiling(inner * min(multiplier, 100.0)))
    }
  }, integer(1L))
  max(required)
}

legacy.finalize <- function(flat.beta, d, nlambda, standardize, multiplier) {
  beta.raw <- matrix(0.0, nrow = d, ncol = nlambda)
  beta.raw[] <- flat.beta[seq_len(d * nlambda)]
  beta <- matrix(0.0, nrow = d, ncol = nlambda)
  beta[seq_len(d), ] <- if (standardize) {
    sweep(beta.raw, 1L, multiplier, `*`)
  } else {
    beta.raw
  }
  list(
    beta = Matrix::Matrix(beta),
    df = as.integer(colSums(beta != 0))
  )
}

refactored.finalize <- function(flat.beta, d, nlambda, standardize,
                                multiplier) {
  beta.raw <- matrix(
    flat.beta[seq_len(d * nlambda)], nrow = d, ncol = nlambda
  )
  beta <- if (standardize) beta.raw * multiplier else beta.raw
  list(
    beta = Matrix::Matrix(beta),
    df = as.integer(colSums(beta != 0))
  )
}

contract.names <- c(
  "beta", "intercept", "lambda", "df", "ite", "nlambda", "gamma",
  "method", "type.gaussian.requested", "type.gaussian", "alg",
  "verbose", "runtime", "fast.mode", "prec", "nulldev", "dev.ratio",
  "family"
)

public.fit <- function(fit.function, arguments) {
  fit <- do.call(fit.function, arguments)
  fit$family <- "gaussian"
  fit
}

assert.fit.parity <- function(arguments, label) {
  old.result <- normalize.fit(public.fit(old.fit, arguments))
  new.result <- normalize.fit(public.fit(new.fit, arguments))
  if (!identical(names(old.result), contract.names) ||
      !identical(names(new.result), contract.names)) {
    stop(sprintf("%s changed the public field order.", label), call. = FALSE)
  }
  if (!identical(class(old.result), class(new.result)) ||
      !identical(dim(old.result$beta), dim(new.result$beta)) ||
      !identical(class(old.result$beta), class(new.result$beta))) {
    stop(sprintf("%s changed a public class or dimension.", label),
         call. = FALSE)
  }
  old.serialized <- serialize(old.result, NULL, version = 3L)
  new.serialized <- serialize(new.result, NULL, version = 3L)
  if (!identical(old.serialized, new.serialized)) {
    stop(sprintf("%s is not serialization-identical.", label), call. = FALSE)
  }
  object.checksum(old.result)
}

set.seed(20260719)
n.oracle <- 80L
d.oracle <- 12L
x.oracle <- matrix(rnorm(n.oracle * d.oracle), nrow = n.oracle)
y.oracle <- x.oracle[, 1L] + 0.8 * x.oracle[, 2L] +
  0.6 * x.oracle[, 3L] + rnorm(n.oracle, sd = 0.1)

oracle.records <- list()
oracle.index <- 0L
for (type.gaussian in c("naive", "covariance")) {
  for (standardize in c(FALSE, TRUE)) {
    for (intercept in c(FALSE, TRUE)) {
      arguments <- list(
        X = x.oracle, Y = y.oracle, lambda = c(0.8, 0.4, 0.2),
        type.gaussian = type.gaussian, standardize = standardize,
        intercept = intercept, prec = 1e-9, max.ite = 10000L
      )
      label <- paste(type.gaussian, standardize, intercept, sep = "/")
      oracle.index <- oracle.index + 1L
      oracle.records[[oracle.index]] <- data.frame(
        oracle = label,
        checksum = assert.fit.parity(arguments, label),
        stringsAsFactors = FALSE
      )
    }
  }
}

x.integer <- cbind(
  constant = rep.int(7L, 48L),
  linear = as.integer(seq_len(48L) - 24L),
  alternating = rep(c(-2L, 3L), length.out = 48L),
  zero = integer(48L)
)
y.integer <- 2.75 + 0.04 * x.integer[, "linear"] -
  0.3 * x.integer[, "alternating"]
for (type.gaussian in c("naive", "covariance")) {
  arguments <- list(
    X = x.integer, Y = y.integer, lambda = 0.25,
    type.gaussian = type.gaussian, standardize = TRUE,
    intercept = TRUE, prec = 1e-9, max.ite = 10000L
  )
  label <- paste0("integer-one-lambda/", type.gaussian)
  oracle.index <- oracle.index + 1L
  oracle.records[[oracle.index]] <- data.frame(
    oracle = label,
    checksum = assert.fit.parity(arguments, label),
    stringsAsFactors = FALSE
  )
}

dfmax.arguments <- list(
  X = x.oracle, Y = y.oracle, nlambda = 40L, lambda.min.ratio = 0.001,
  dfmax = 1L, type.gaussian = "naive", standardize = TRUE,
  intercept = TRUE, prec = 1e-8, max.ite = 10000L
)
oracle.index <- oracle.index + 1L
oracle.records[[oracle.index]] <- data.frame(
  oracle = "dfmax-prefix",
  checksum = assert.fit.parity(dfmax.arguments, "dfmax-prefix"),
  stringsAsFactors = FALSE
)
oracle.table <- do.call(rbind, oracle.records)

set.seed(20260720)
kernel.cases <- list(
  small_d_unstandardized = list(d = 64L, nlambda = 100L,
                                standardize = FALSE, inner = 25L),
  small_d_standardized = list(d = 64L, nlambda = 100L,
                              standardize = TRUE, inner = 25L),
  large_d_unstandardized = list(d = 5000L, nlambda = 100L,
                                standardize = FALSE, inner = 5L),
  large_d_standardized = list(d = 5000L, nlambda = 100L,
                              standardize = TRUE, inner = 5L)
)

rows <- list()
row.index <- 0L
for (case.name in names(kernel.cases)) {
  case <- kernel.cases[[case.name]]
  flat.beta <- rnorm(case$d * case$nlambda)
  flat.beta[seq.int(1L, length(flat.beta), by = 11L)] <- 0.0
  multiplier <- if (case$standardize) {
    runif(case$d, min = 0.25, max = 4.0)
  } else {
    rep(1.0, case$d)
  }
  operations <- list(
    old = function() legacy.finalize(
      flat.beta, case$d, case$nlambda, case$standardize, multiplier
    ),
    new = function() refactored.finalize(
      flat.beta, case$d, case$nlambda, case$standardize, multiplier
    )
  )

  invisible(operations$old())
  invisible(operations$new())
  case$inner <- calibrate.inner(operations, case$inner)
  expected <- serialize(operations$old(), NULL, version = 3L)
  if (!identical(expected, serialize(operations$new(), NULL, version = 3L))) {
    stop(sprintf("Kernel oracle failed for %s.", case.name), call. = FALSE)
  }

  for (repetition in seq_len(repetitions)) {
    order <- if (repetition %% 2L == 1L) c("old", "new") else c("new", "old")
    for (position in seq_along(order)) {
      implementation <- order[[position]]
      measured <- profile.once(
        operations[[implementation]], inner = case$inner
      )
      row.index <- row.index + 1L
      rows[[row.index]] <- data.frame(
        benchmark = "finalization_kernel",
        case = case.name,
        repetition = repetition,
        position = position,
        implementation = implementation,
        elapsed_seconds = measured$elapsed.seconds,
        batch_elapsed_seconds = measured$batch.elapsed.seconds,
        inner_iterations = measured$inner,
        allocated_bytes = measured$allocated.bytes,
        checksum = object.checksum(measured$result),
        old_source_md5 = source.hash[["old"]],
        new_source_md5 = source.hash[["new"]],
        native_md5 = native.hash,
        native_path = provenance$native.path,
        r_version = provenance$r.version,
        matrix_version = provenance$matrix.version,
        blas = provenance$blas,
        cpu = provenance$cpu,
        platform = provenance$platform,
        thread_settings = provenance$threads,
        benchmark_script_md5 = provenance$script.hash,
        stringsAsFactors = FALSE
      )
    }
  }
}

set.seed(20260721)
tall.x <- matrix(rnorm(2000L * 40L), nrow = 2000L)
tall.y <- 0.5 + tall.x[, 1L] - 0.7 * tall.x[, 2L] +
  rnorm(nrow(tall.x), sd = 0.25)
wide.x <- matrix(rnorm(80L * 400L), nrow = 80L)
wide.y <- wide.x[, 1L] + 0.5 * wide.x[, 2L] +
  rnorm(nrow(wide.x), sd = 0.25)
covariance.x <- matrix(rnorm(600L * 60L), nrow = 600L)
covariance.y <- 1.0 + covariance.x[, 1L] -
  0.4 * covariance.x[, 2L] + rnorm(nrow(covariance.x), sd = 0.25)
benchmark.lambda <- exp(seq(log(1.0), log(0.2), length.out = 20L))
fit.cases <- list(
  tall_naive_standardized = list(
    X = tall.x, Y = tall.y, lambda = benchmark.lambda,
    type.gaussian = "naive", standardize = TRUE, intercept = TRUE,
    prec = 1e-7, max.ite = 10000L
  ),
  wide_naive_unstandardized = list(
    X = wide.x, Y = wide.y, lambda = benchmark.lambda,
    type.gaussian = "naive", standardize = FALSE, intercept = FALSE,
    prec = 1e-7, max.ite = 10000L
  ),
  tall_covariance_standardized = list(
    X = covariance.x, Y = covariance.y, lambda = benchmark.lambda,
    type.gaussian = "covariance", standardize = TRUE, intercept = TRUE,
    prec = 1e-7, max.ite = 10000L
  )
)

for (case.name in names(fit.cases)) {
  fit.arguments <- fit.cases[[case.name]]
  expected.checksum <- assert.fit.parity(fit.arguments, case.name)
  fit.functions <- list(old = old.fit, new = new.fit)
  operations <- list(
    old = function() public.fit(fit.functions$old, fit.arguments),
    new = function() public.fit(fit.functions$new, fit.arguments)
  )
  invisible(operations$old())
  invisible(operations$new())
  fit.inner <- calibrate.inner(operations, initial = 20L)

  for (repetition in seq_len(repetitions)) {
    order <- if (repetition %% 2L == 1L) c("old", "new") else c("new", "old")
    for (position in seq_along(order)) {
      implementation <- order[[position]]
      measured <- profile.once(operations[[implementation]], inner = fit.inner)
      normalized <- normalize.fit(measured$result)
      checksum <- object.checksum(normalized)
      if (!identical(checksum, expected.checksum)) {
        stop(sprintf("Fit checksum changed for %s/%s.",
                     case.name, implementation), call. = FALSE)
      }
      row.index <- row.index + 1L
      rows[[row.index]] <- data.frame(
        benchmark = "full_wrapper_fit",
        case = case.name,
        repetition = repetition,
        position = position,
        implementation = implementation,
        elapsed_seconds = measured$elapsed.seconds,
        batch_elapsed_seconds = measured$batch.elapsed.seconds,
        inner_iterations = measured$inner,
        allocated_bytes = measured$allocated.bytes,
        checksum = checksum,
        old_source_md5 = source.hash[["old"]],
        new_source_md5 = source.hash[["new"]],
        native_md5 = native.hash,
        native_path = provenance$native.path,
        r_version = provenance$r.version,
        matrix_version = provenance$matrix.version,
        blas = provenance$blas,
        cpu = provenance$cpu,
        platform = provenance$platform,
        thread_settings = provenance$threads,
        benchmark_script_md5 = provenance$script.hash,
        stringsAsFactors = FALSE
      )
    }
  }
}

results <- do.call(rbind, rows)
if (any(results$batch_elapsed_seconds < 0.1)) {
  stop("A benchmark batch completed in less than 0.1 seconds.", call. = FALSE)
}
output.directory <- dirname(output.csv)
if (!dir.exists(output.directory)) {
  stop(sprintf("Output directory does not exist: %s", output.directory),
       call. = FALSE)
}
write.csv(results, output.csv, row.names = FALSE, quote = TRUE)

cat("Serialization oracles:\n")
print(oracle.table, row.names = FALSE)
cat("\nMedian benchmark results:\n")
summary <- aggregate(
  cbind(elapsed_seconds, allocated_bytes) ~ case + implementation,
  data = results, FUN = median
)
print(summary, row.names = FALSE)
cat(sprintf("\nRaw results written to %s\n", normalizePath(output.csv)))
