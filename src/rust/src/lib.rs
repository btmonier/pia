//! Rust core of the `pia` R package: pixelated image abstraction after
//! Gerstner et al. (2012). The R wrappers for the `#[extendr]` functions are
//! generated into `R/extendr-wrappers.R`; the user facing tidyverse API lives
//! on the R side.

use extendr_api::prelude::*;

/// Result type of the `#[extendr]` entry points.
type Result<T> = std::result::Result<T, Error>;

pub mod bilateral;
pub mod color;
pub mod mcda;
pub mod pixelate;
pub mod slic;

use color::Lab;
use mcda::AnnealParams;
use pixelate::{pixelate, PixelateParams};

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Other(msg.into()))
}

/// Convert an `n x 3` numeric matrix into a vector of LAB colors.
fn matrix_to_lab(m: &RMatrix<f64>, what: &str) -> Result<Vec<Lab>> {
    if m.ncols() != 3 {
        return err(format!("`{what}` must have exactly 3 columns (L, a, b)."));
    }
    let n = m.nrows();
    let data = m.data();
    Ok((0..n).map(|i| [data[i], data[i + n], data[i + 2 * n]]).collect())
}

/// Convert a vector of LAB colors into an `n x 3` numeric matrix.
fn lab_to_matrix(colors: &[Lab]) -> RMatrix<f64> {
    RMatrix::new_matrix(colors.len(), 3, |r, c| colors[r][c])
}

fn to_usize(v: i32, what: &str) -> Result<usize> {
    if v < 1 {
        err(format!("`{what}` must be a positive integer."))
    } else {
        Ok(v as usize)
    }
}

/// Run the full pixelated image abstraction pipeline.
///
/// Internal entry point used by `pia_pixelate()`.
///
/// @param lab Numeric matrix with `w_in * h_in` rows and 3 columns holding
///   CIELAB colors in row-major pixel order (`row = y * w_in + x`, 0-based).
/// @param w_in,h_in Input width and height in pixels.
/// @param w_out,h_out Output width and height in pixels.
/// @param palette_size Maximum number of palette colors `K`.
/// @param m SLIC positional weight.
/// @param alpha Temperature reduction factor.
/// @param t_final Final temperature.
/// @param laplacian Laplacian smoothing weight for superpixel centers.
/// @param bilateral Whether to bilaterally filter superpixel colors.
/// @param bilateral_sigma_s,bilateral_sigma_r Spatial and range sigmas of the
///   bilateral filter.
/// @param importance Optional numeric vector of per-pixel weights, or `NULL`.
/// @param max_iter Maximum number of iterations.
/// @param eps_palette,eps_cluster,perturbation Annealing constants.
/// @return A list with `assignment`, `palette_id`, `palette`, `centers`,
///   `superpixel_colors`, `iterations`, `temperature`,
///   `initial_temperature` and `converged`.
/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn pia_core(
    lab: RMatrix<f64>,
    w_in: i32,
    h_in: i32,
    w_out: i32,
    h_out: i32,
    palette_size: i32,
    m: f64,
    alpha: f64,
    t_final: f64,
    laplacian: f64,
    bilateral: bool,
    bilateral_sigma_s: f64,
    bilateral_sigma_r: f64,
    importance: Option<Vec<f64>>,
    max_iter: i32,
    eps_palette: f64,
    eps_cluster: f64,
    perturbation: f64,
) -> Result<List> {
    let lab = matrix_to_lab(&lab, "lab")?;
    let w_in = to_usize(w_in, "w_in")?;
    let h_in = to_usize(h_in, "h_in")?;
    let w_out = to_usize(w_out, "w_out")?;
    let h_out = to_usize(h_out, "h_out")?;
    let palette_size = to_usize(palette_size, "palette_size")?;
    let max_iter = to_usize(max_iter, "max_iter")?;

    if lab.len() != w_in * h_in {
        return err("`lab` must have `w_in * h_in` rows.");
    }
    if let Some(imp) = &importance {
        if imp.len() != lab.len() {
            return err("`importance` must have one value per input pixel.");
        }
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return err("`alpha` must be strictly between 0 and 1.");
    }
    if !(t_final > 0.0) {
        return err("`t_final` must be positive.");
    }

    let params = PixelateParams {
        w_out,
        h_out,
        palette_size,
        m,
        t_final,
        laplacian,
        bilateral,
        bilateral_sigma_s,
        bilateral_sigma_r,
        max_iter,
        anneal: AnnealParams {
            alpha,
            eps_palette,
            eps_cluster,
            perturbation,
        },
    };

    let res = pixelate(&lab, w_in, h_in, importance.as_deref(), &params, None);

    let assignment: Vec<i32> = res.assignment.iter().map(|&s| s as i32 + 1).collect();
    let palette_id: Vec<i32> = res.palette_id.iter().map(|&k| k as i32 + 1).collect();
    let centers = RMatrix::new_matrix(res.superpixels.len(), 2, |r, c| {
        if c == 0 {
            res.superpixels[r].cx
        } else {
            res.superpixels[r].cy
        }
    });
    let counts: Vec<i32> = res.superpixels.iter().map(|s| s.count as i32).collect();

    Ok(list!(
        assignment = assignment,
        palette_id = palette_id,
        palette = lab_to_matrix(&res.palette),
        centers = centers,
        counts = counts,
        superpixel_colors = lab_to_matrix(&res.smoothed_colors),
        iterations = res.iterations as i32,
        temperature = res.temperature,
        initial_temperature = res.initial_temperature,
        converged = res.converged
    ))
}

/// Critical temperature of a set of CIELAB colors.
///
/// Twice the variance along the major principal component axis (Rose 1998).
///
/// @param lab Numeric matrix with 3 columns.
/// @param weights Optional non-negative weights, one per row.
/// @noRd
#[extendr]
fn pia_critical_temperature(lab: RMatrix<f64>, weights: Option<Vec<f64>>) -> Result<f64> {
    let lab = matrix_to_lab(&lab, "lab")?;
    let w = match weights {
        Some(w) => {
            if w.len() != lab.len() {
                return err("`weights` must have one value per row of `lab`.");
            }
            w
        }
        None => vec![1.0; lab.len()],
    };
    Ok(color::critical_temperature(&lab, &w))
}

/// One SLIC assignment step.
///
/// @param lab Row-major `w_in * h_in` by 3 matrix of CIELAB colors.
/// @param w_in,h_in Input dimensions.
/// @param w_out,h_out Superpixel grid dimensions.
/// @param center_x,center_y Superpixel centers (0-based input coordinates).
/// @param colors `w_out * h_out` by 3 matrix of representative colors.
/// @param m Positional weight.
/// @return Integer vector of 1-based superpixel indices per input pixel.
/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn pia_slic_assign(
    lab: RMatrix<f64>,
    w_in: i32,
    h_in: i32,
    w_out: i32,
    h_out: i32,
    center_x: Vec<f64>,
    center_y: Vec<f64>,
    colors: RMatrix<f64>,
    m: f64,
) -> Result<Vec<i32>> {
    let lab = matrix_to_lab(&lab, "lab")?;
    let colors = matrix_to_lab(&colors, "colors")?;
    let w_in = to_usize(w_in, "w_in")?;
    let h_in = to_usize(h_in, "h_in")?;
    let w_out = to_usize(w_out, "w_out")?;
    let h_out = to_usize(h_out, "h_out")?;
    let n_out = w_out * h_out;
    if lab.len() != w_in * h_in {
        return err("`lab` must have `w_in * h_in` rows.");
    }
    if center_x.len() != n_out || center_y.len() != n_out || colors.len() != n_out {
        return err("Centers and colors must have `w_out * h_out` entries.");
    }
    let mut sps = slic::init_grid(w_in, h_in, w_out, h_out, [0.0; 3]);
    for (s, sp) in sps.iter_mut().enumerate() {
        sp.cx = center_x[s];
        sp.cy = center_y[s];
        sp.color = colors[s];
    }
    let a = slic::assign(&lab, w_in, h_in, &sps, w_out, h_out, m);
    Ok(a.into_iter().map(|s| s as i32 + 1).collect())
}

/// Conditional association probabilities of MCDA (Equation 2).
///
/// @param colors `N` by 3 matrix of superpixel colors.
/// @param palette `J` by 3 matrix of cluster colors.
/// @param palette_prob Marginal probability `P(c_j)` of each cluster.
/// @param temperature Annealing temperature.
/// @return `N` by `J` matrix whose rows sum to one.
/// @noRd
#[extendr]
fn pia_mcda_associate(
    colors: RMatrix<f64>,
    palette: RMatrix<f64>,
    palette_prob: Vec<f64>,
    temperature: f64,
) -> Result<RMatrix<f64>> {
    let colors = matrix_to_lab(&colors, "colors")?;
    let palette = matrix_to_lab(&palette, "palette")?;
    if palette_prob.len() != palette.len() {
        return err("`palette_prob` must have one value per palette row.");
    }
    if !(temperature > 0.0) {
        return err("`temperature` must be positive.");
    }
    let owner: Vec<usize> = (0..palette.len()).collect();
    let assoc = mcda::conditional_probabilities(&colors, &palette, &palette_prob, temperature, &owner);
    Ok(RMatrix::new_matrix(assoc.n_superpixels, assoc.n_clusters, |r, c| assoc.get(r, c)))
}

/// Bilateral filter of a row-major grid of CIELAB colors.
///
/// @param colors `w * h` by 3 matrix.
/// @param w,h Grid dimensions.
/// @param sigma_s,sigma_r Spatial and range sigmas.
/// @noRd
#[extendr]
fn pia_bilateral(colors: RMatrix<f64>, w: i32, h: i32, sigma_s: f64, sigma_r: f64) -> Result<RMatrix<f64>> {
    let colors = matrix_to_lab(&colors, "colors")?;
    let w = to_usize(w, "w")?;
    let h = to_usize(h, "h")?;
    if colors.len() != w * h {
        return err("`colors` must have `w * h` rows.");
    }
    let out = bilateral::bilateral(&colors, w, h, sigma_s, sigma_r);
    Ok(lab_to_matrix(&out))
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod pia;
    fn pia_core;
    fn pia_critical_temperature;
    fn pia_slic_assign;
    fn pia_mcda_associate;
    fn pia_bilateral;
}
