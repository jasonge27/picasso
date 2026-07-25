args <- commandArgs(trailingOnly = TRUE)

load.implementation <- function(source.dir) {
  environment <- new.env(parent = asNamespace("picasso"))
  for (file in c(
    "picasso_utils.R", "picasso.multinomial.R",
    "assess.picasso.R", "cv.picasso.R"
  )) {
    sys.source(file.path(source.dir, file), environment)
  }
  environment
}

make.model <- function(d, classes, nlambda, lambda = NULL) {
  K <- length(classes)
  beta <- lapply(seq_len(K), function(class) {
    matrix(
      0.2 * sin(seq_len(d * nlambda) * (0.011 + 0.002 * class)),
      d, nlambda
    )
  })
  intercept <- lapply(seq_len(K), function(class) {
    0.15 * cos(seq_len(nlambda) * (0.07 + 0.01 * class))
  })
  if (is.null(lambda)) lambda <- seq(1, 0.2, length.out = nlambda)
  structure(list(
    beta = beta,
    intercept = intercept,
    lambda = lambda,
    nlambda = nlambda,
    K = K,
    levels = classes,
    family = "multinomial",
    status = "completed",
    status.code = 0L,
    path.early.stopped = FALSE,
    fast.mode = FALSE,
    prec = 1e-4
  ), class = "multinomial")
}

profile.memory <- function(expression) {
  path <- tempfile("picasso-rprofmem-")
  on.exit(unlink(path), add = TRUE)
  gc()
  Rprofmem(path)
  profiling <- TRUE
  on.exit(if (profiling) Rprofmem(NULL), add = TRUE)
  timing <- system.time(result <- force(expression))[["elapsed"]]
  Rprofmem(NULL)
  profiling <- FALSE
  records <- readLines(path, warn = FALSE)
  sizes <- suppressWarnings(as.numeric(sub(" .*", "", records)))
  list(
    result = result,
    elapsed = unname(timing),
    allocated = sum(sizes, na.rm = TRUE)
  )
}

checksum <- function(value) {
  path <- tempfile("picasso-stream-result-", fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(value, path, version = 3L)
  unname(tools::md5sum(path))
}

run.child <- function(source.dir, case, output) {
  environment <- load.implementation(source.dir)
  classes <- paste0("class", seq_len(3L))
  d <- 1L
  if (case == "assess") {
    nlambda <- 60L
    n <- 60000L
  } else if (startsWith(case, "cv_")) {
    nlambda <- 25L
    n <- 24000L
  } else {
    nlambda <- 30L
    n <- 20000L
  }
  x <- matrix(sin(seq_len(n * d) * 0.013), n, d)
  y <- factor(
    rep(classes, length.out = n),
    levels = c(rev(classes), "unused")
  )
  object <- make.model(d, classes, nlambda)

  if (startsWith(case, "cv_")) {
    environment$picasso <- function(X, Y, ..., lambda = NULL,
                                    nlambda = NULL,
                                    lambda.min.ratio = NULL) {
      fitted.levels <- levels(droplevels(as.factor(Y)))
      fitted.length <- if (is.null(lambda)) as.integer(nlambda) else length(lambda)
      make.model(ncol(X), fitted.levels, fitted.length, lambda)
    }
    foldid <- integer(n)
    for (class in classes) {
      index <- which(y == class)
      foldid[index] <- rep(seq_len(3L), length.out = length(index))
    }
    measure <- sub("^cv_", "", case)
    expression <- quote(environment$cv.picasso(
      x, y, family = "multinomial", foldid = foldid,
      nlambda = nlambda, lambda.min.ratio = 0.2,
      type.measure = measure, prec = 1e-4, max.ite = 100L
    ))
  } else if (case == "assess") {
    expression <- quote(environment$assess.picasso(object, x, y))
  } else if (case == "confusion") {
    expression <- quote(environment$confusion.picasso(
      object, x, y, lambda.idx = seq_len(nlambda)
    ))
  } else if (case == "predict_class") {
    expression <- quote(environment$predict.multinomial(
      object, x, lambda.idx = seq_len(nlambda), type = "class"
    ))
  } else {
    stop("Unknown benchmark case: ", case)
  }

  measured <- profile.memory(eval(expression))
  saveRDS(list(
    elapsed = measured$elapsed,
    allocated = measured$allocated,
    checksum = checksum(measured$result),
    observations = n,
    classes = length(classes),
    lambdas = nlambda
  ), output, version = 3L)
}

if (length(args) > 0L && identical(args[[1L]], "--child")) {
  if (length(args) != 4L)
    stop("Child usage: --child SOURCE_DIR CASE OUTPUT")
  run.child(args[[2L]], args[[3L]], args[[4L]])
  quit(save = "no", status = 0L)
}

if (length(args) < 4L || length(args) > 5L) {
  stop(paste(
    "Usage: Rscript r_multinomial_streaming_benchmark.R",
    "OLD_SOURCE_DIR NEW_SOURCE_DIR RESULTS_CSV REPORT_MD [REPETITIONS]"
  ))
}

old.dir <- normalizePath(args[[1L]], mustWork = TRUE)
new.dir <- normalizePath(args[[2L]], mustWork = TRUE)
results.path <- args[[3L]]
report.path <- args[[4L]]
repetitions <- if (length(args) == 5L) as.integer(args[[5L]]) else 7L
if (is.na(repetitions) || repetitions < 7L)
  stop("Use at least seven repetitions for the final benchmark.")

script.path <- normalizePath(
  sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[[1L]]),
  mustWork = TRUE
)
rscript <- file.path(R.home("bin"), "Rscript")
cases <- c("assess", "confusion", "predict_class", "cv_class", "cv_deviance")

source.hash <- function(directory) {
  files <- file.path(directory, c(
    "picasso_utils.R", "picasso.multinomial.R",
    "assess.picasso.R", "cv.picasso.R"
  ))
  paste(paste(basename(files), unname(tools::md5sum(files)), sep = "="),
        collapse = ";")
}

hardware <- suppressWarnings(tryCatch(
  system2("system_profiler", "SPHardwareDataType", stdout = TRUE),
  error = function(error) character()
))
chip <- grep("^[[:space:]]*Chip:", hardware, value = TRUE)
cpu <- if (length(chip) == 1L) trimws(sub("^[^:]+:", "", chip)) else {
  paste(Sys.info()[c("sysname", "machine")], collapse = " ")
}
thread.variables <- c(
  "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
  "MKL_NUM_THREADS", "BLIS_NUM_THREADS"
)
threads <- paste(
  paste0(thread.variables, "=", Sys.getenv(thread.variables, unset = "<unset>")),
  collapse = ";"
)
metadata <- list(
  script_hash = unname(tools::md5sum(script.path)),
  old_source_hash = source.hash(old.dir),
  new_source_hash = source.hash(new.dir),
  r_version = R.version.string,
  platform = R.version$platform,
  matrix_version = as.character(utils::packageVersion("Matrix")),
  blas = unname(extSoftVersion()[["BLAS"]]),
  lapack = unname(La_library()),
  cpu = paste(cpu, collapse = " "),
  threads = threads
)

run.fresh <- function(source.dir, case) {
  child.output <- tempfile("picasso-stream-child-", fileext = ".rds")
  stdout <- tempfile("picasso-stream-stdout-")
  stderr <- tempfile("picasso-stream-stderr-")
  on.exit(unlink(c(child.output, stdout, stderr)), add = TRUE)
  status <- system2(
    "/usr/bin/time",
    c(
      "-l", shQuote(rscript), shQuote(script.path), "--child",
      shQuote(source.dir), case, shQuote(child.output)
    ),
    stdout = stdout,
    stderr = stderr
  )
  if (status != 0L || !file.exists(child.output)) {
    stop(paste(
      "Fresh-process benchmark failed:",
      paste(readLines(stderr, warn = FALSE), collapse = "\n")
    ))
  }
  timing.output <- readLines(stderr, warn = FALSE)
  rss.line <- grep("maximum resident set size", timing.output, value = TRUE)
  if (length(rss.line) != 1L)
    stop("Could not parse fresh-process maximum resident set size.")
  maximum.rss <- suppressWarnings(as.numeric(
    trimws(sub("maximum resident set size.*$", "", rss.line))
  ))
  if (!is.finite(maximum.rss))
    stop("Fresh-process maximum resident set size was not numeric.")
  c(readRDS(child.output), list(maximum_rss = maximum.rss))
}

records <- vector("list", length(cases) * repetitions * 2L)
position <- 0L
for (case in cases) {
  for (repetition in seq_len(repetitions)) {
    order <- if (repetition %% 2L == 1L) c("old", "new") else c("new", "old")
    for (implementation in order) {
      position <- position + 1L
      source.dir <- if (implementation == "old") old.dir else new.dir
      value <- run.fresh(source.dir, case)
      records[[position]] <- data.frame(
        case = case,
        repetition = repetition,
        implementation = implementation,
        elapsed_seconds = value$elapsed,
        allocated_bytes = value$allocated,
        maximum_rss_bytes = value$maximum_rss,
        checksum = value$checksum,
        observations = value$observations,
        classes = value$classes,
        lambdas = value$lambdas,
        stringsAsFactors = FALSE
      )
    }
  }
}
results <- do.call(rbind, records)
for (name in names(metadata)) results[[name]] <- metadata[[name]]
utils::write.csv(results, results.path, row.names = FALSE)

summaries <- lapply(cases, function(case) {
  subset <- results[results$case == case, ]
  old <- subset[subset$implementation == "old", ]
  new <- subset[subset$implementation == "new", ]
  matrix.observations <- if (startsWith(case, "cv_")) {
    old$observations[[1L]] / 3
  } else {
    old$observations[[1L]]
  }
  matrix.bytes <- 8 * matrix.observations * old$classes[[1L]]
  old.multiplier <- if (case == "predict_class") {
    2
  } else {
    old$lambdas[[1L]] + 1
  }
  new.multiplier <- if (case == "cv_deviance") 1 else 2
  data.frame(
    case = case,
    old_time = median(old$elapsed_seconds),
    new_time = median(new$elapsed_seconds),
    time_change = median(new$elapsed_seconds) / median(old$elapsed_seconds) - 1,
    old_alloc = median(old$allocated_bytes),
    new_alloc = median(new$allocated_bytes),
    alloc_change = median(new$allocated_bytes) / median(old$allocated_bytes) - 1,
    old_rss = median(old$maximum_rss_bytes),
    new_rss = median(new$maximum_rss_bytes),
    rss_change = median(new$maximum_rss_bytes) / median(old$maximum_rss_bytes) - 1,
    old_live_payload = old.multiplier * matrix.bytes,
    new_live_payload = new.multiplier * matrix.bytes,
    equivalent = length(unique(subset$checksum)) == 1L
  )
})
summary <- do.call(rbind, summaries)

format.bytes <- function(value) sprintf("%.1f MiB", value / 1024^2)
rows <- vapply(seq_len(nrow(summary)), function(index) {
  row <- summary[index, ]
  sprintf(
    "| %s | %.4f | %.4f | %+.1f%% | %s | %s | %+.1f%% | %s | %s | %+.1f%% | %s |",
    row$case, row$old_time, row$new_time, 100 * row$time_change,
    format.bytes(row$old_alloc), format.bytes(row$new_alloc),
    100 * row$alloc_change,
    format.bytes(row$old_rss), format.bytes(row$new_rss),
    100 * row$rss_change, if (row$equivalent) "yes" else "NO"
  )
}, character(1))

runtime.ok <- all(summary$time_change <= 0.05)
equivalent <- all(summary$equivalent)
report <- c(
  "# R Multinomial Streaming Benchmark",
  "",
  sprintf("Seven-run minimum: %d alternating fresh-process runs per implementation and case.", repetitions),
  "`Rprofmem` is cumulative allocation, not peak memory. Maximum RSS is the fresh-process high-water mark reported by macOS `/usr/bin/time -l`; it includes the R runtime, inputs, and fitted object in both implementations.",
  "",
  "| Case | Old time (s) | New time (s) | Time | Old allocation | New allocation | Allocation | Old max RSS | New max RSS | Max RSS | Exact output |",
  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
  rows,
  "",
  "The retained-path change is clearer in the exact core-array live payload (logits plus the simultaneously used probability matrix where applicable):",
  "",
  "| Case | Old live payload | New live payload | Change |",
  "|---|---:|---:|---:|",
  vapply(seq_len(nrow(summary)), function(index) {
    row <- summary[index, ]
    sprintf(
      "| %s | %s | %s | %+.1f%% |",
      row$case, format.bytes(row$old_live_payload),
      format.bytes(row$new_live_payload),
      100 * (row$new_live_payload / row$old_live_payload - 1)
    )
  }, character(1)),
  "",
  sprintf("Decision: %s (output equivalence: %s; no median runtime regression above 5%%: %s).",
          if (equivalent && runtime.ok) "KEEP" else "REVIEW/ROLL BACK",
          equivalent, runtime.ok),
  "",
  "## Compatibility audit",
  "",
  paste0(
    "Classification still applies the historical softmax to each streamed ",
    "logits matrix before first-tie selection. For logits ",
    "`[0, f * .Machine$double.eps, -2]`, direct logits select class 2 for ",
    "`f = 0.125, 0.25, 0.5, 1, 2`, while softmax-first selects classes ",
    "`1, 1, 2, 2, 2`; tests lock this near-tie boundary."
  ),
  paste0(
    "NaN and positive-infinite logits retain the old controlled softmax ",
    "error. An isolated negative-infinite class logit remains accepted as ",
    "zero probability by softmax (the legacy behavior), while multinomial ",
    "NLL assessment continues to require finite logits."
  ),
  "",
  "## Reproducibility",
  "",
  paste0("- Script MD5: `", metadata$script_hash, "`"),
  paste0("- Old sources: `", metadata$old_source_hash, "`"),
  paste0("- New sources: `", metadata$new_source_hash, "`"),
  paste0("- R: ", metadata$r_version, " (", metadata$platform, ")"),
  paste0("- Matrix: ", metadata$matrix_version),
  paste0("- BLAS: `", metadata$blas, "`"),
  paste0("- LAPACK: `", metadata$lapack, "`"),
  paste0("- CPU: ", metadata$cpu),
  paste0("- Thread environment: `", metadata$threads, "`")
)
writeLines(report, report.path)
print(summary)
