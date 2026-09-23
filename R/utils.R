# Internal helpers: argument checking and color conversion -------------------

check_number <- function(x, arg = rlang::caller_arg(x), min = -Inf, max = Inf,
                         allow_null = FALSE, call = rlang::caller_env()) {
  if (is.null(x) && allow_null) {
    return(invisible(NULL))
  }
  if (!is.numeric(x) || length(x) != 1L || is.na(x)) {
    cli::cli_abort("{.arg {arg}} must be a single number, not {.obj_type_friendly {x}}.", call = call)
  }
  if (x < min || x > max) {
    cli::cli_abort("{.arg {arg}} must be between {min} and {max}, not {x}.", call = call)
  }
  invisible(x)
}

check_count <- function(x, arg = rlang::caller_arg(x), min = 1L,
                        allow_null = FALSE, call = rlang::caller_env()) {
  if (is.null(x) && allow_null) {
    return(invisible(NULL))
  }
  if (!is_scalar_integerish(x) || is.na(x) || x < min) {
    cli::cli_abort(
      "{.arg {arg}} must be a whole number of at least {min}, not {.obj_type_friendly {x}}.",
      call = call
    )
  }
  invisible(as.integer(x))
}

check_flag <- function(x, arg = rlang::caller_arg(x), call = rlang::caller_env()) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    cli::cli_abort("{.arg {arg}} must be `TRUE` or `FALSE`.", call = call)
  }
  invisible(x)
}

clamp01 <- function(x) {
  pmin(pmax(x, 0), 1)
}

#' Convert between sRGB (in `[0, 1]`) and CIELAB
#'
#' Thin wrappers around [farver::convert_colour()] that use the `[0, 1]`
#' convention for RGB values adopted throughout pia.
#'
#' @param rgb A numeric matrix with three columns (`r`, `g`, `b`) in `[0, 1]`.
#' @param lab A numeric matrix with three columns (`L`, `a`, `b`).
#' @return `rgb_to_lab()` returns a numeric matrix of CIELAB values,
#'   `lab_to_rgb()` a numeric matrix of RGB values clamped to `[0, 1]`, and
#'   `lab_to_hex()` a character vector of hex color strings.
#' @examples
#' lab <- rgb_to_lab(matrix(c(1, 0, 0), ncol = 3))
#' lab_to_rgb(lab)
#' lab_to_hex(lab)
#' @name color_conversion
NULL

#' @rdname color_conversion
#' @export
rgb_to_lab <- function(rgb) {
  rgb <- as_color_matrix(rgb)
  out <- farver::convert_colour(rgb * 255, from = "rgb", to = "lab")
  colnames(out) <- c("l", "a", "b")
  out
}

#' @rdname color_conversion
#' @export
lab_to_rgb <- function(lab) {
  lab <- as_color_matrix(lab)
  out <- farver::convert_colour(lab, from = "lab", to = "rgb") / 255
  out <- clamp01(out)
  colnames(out) <- c("r", "g", "b")
  out
}

#' @rdname color_conversion
#' @export
lab_to_hex <- function(lab) {
  rgb <- lab_to_rgb(lab)
  farver::encode_colour(rgb * 255, from = "rgb")
}

as_color_matrix <- function(x, call = rlang::caller_env()) {
  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }
  if (!is.matrix(x)) {
    x <- matrix(as.numeric(x), ncol = 3)
  }
  if (ncol(x) != 3L) {
    cli::cli_abort("Color matrices must have exactly 3 columns.", call = call)
  }
  storage.mode(x) <- "double"
  unname(x)
}

#' Saturate CIELAB colors
#'
#' Multiplies the `a` and `b` channels by `beta`, the palette saturation
#' post-processing step of Section 4.4 of the paper.
#'
#' @param lab A numeric matrix of CIELAB colors.
#' @param beta Saturation factor; `1` leaves colors unchanged.
#' @return A numeric matrix of the same shape.
#' @examples
#' saturate_lab(matrix(c(50, 20, -20), ncol = 3), beta = 1.1)
#' @export
saturate_lab <- function(lab, beta = 1.1) {
  check_number(beta, min = 0)
  lab <- as_color_matrix(lab)
  lab[, 2:3] <- lab[, 2:3] * beta
  colnames(lab) <- c("l", "a", "b")
  lab
}
