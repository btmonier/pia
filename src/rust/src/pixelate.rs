//! Driver for Algorithm 1 of Gerstner et al. 2012.
//!
//! ```text
//! initialize superpixels, palette and temperature T
//! while (T > T_f)
//!     refine superpixels with one step of modified SLIC
//!     associate superpixels to colors in the palette
//!     refine colors in the palette
//!     if (palette converged)
//!         reduce temperature T = alpha T
//!         expand palette
//! post-process
//! ```

use crate::bilateral::bilateral;
use crate::color::{covariance, critical_temperature, mean, principal_axis, Lab};
use crate::mcda::{AnnealParams, Palette};
use crate::slic::{assign, init_grid, laplacian_smooth, update, Superpixel};

/// User facing parameters of the pixelation.
#[derive(Clone, Debug)]
pub struct PixelateParams {
    pub w_out: usize,
    pub h_out: usize,
    pub palette_size: usize,
    /// SLIC positional weight `m`.
    pub m: f64,
    /// Final temperature `T_f`.
    pub t_final: f64,
    /// Laplacian smoothing weight for superpixel centers (0 disables).
    pub laplacian: f64,
    /// Whether to bilaterally filter superpixel colors.
    pub bilateral: bool,
    pub bilateral_sigma_s: f64,
    pub bilateral_sigma_r: f64,
    pub max_iter: usize,
    pub anneal: AnnealParams,
}

/// Output of the pixelation.
#[derive(Clone, Debug)]
pub struct PixelateResult {
    /// Superpixel index (0-based) for every input pixel, row-major.
    pub assignment: Vec<usize>,
    /// Palette color index (0-based) for every superpixel / output pixel.
    pub palette_id: Vec<usize>,
    /// Final palette colors in CIELAB.
    pub palette: Vec<Lab>,
    /// Final superpixel state.
    pub superpixels: Vec<Superpixel>,
    /// Filtered superpixel colors `m_s'`.
    pub smoothed_colors: Vec<Lab>,
    pub iterations: usize,
    pub temperature: f64,
    pub initial_temperature: f64,
    pub converged: bool,
}

/// Normalized prior `P(p_s)` from the superpixels' mean importance.
fn superpixel_prior(sps: &[Superpixel], has_importance: bool) -> Vec<f64> {
    let mut w: Vec<f64> = sps
        .iter()
        .map(|s| {
            if s.count == 0 {
                0.0
            } else if has_importance {
                s.importance.max(0.0)
            } else {
                1.0
            }
        })
        .collect();
    let total: f64 = w.iter().sum();
    if total <= 0.0 {
        let n = sps.len() as f64;
        return vec![1.0 / n; sps.len()];
    }
    for v in w.iter_mut() {
        *v /= total;
    }
    w
}

/// Run the full pixelation. `lab` is row-major (`index = y * w_in + x`) and
/// `importance`, when given, has one weight in `[0, 1]` per input pixel.
pub fn pixelate(
    lab: &[Lab],
    w_in: usize,
    h_in: usize,
    importance: Option<&[f64]>,
    params: &PixelateParams,
    mut progress: Option<&mut dyn FnMut(usize, f64, usize)>,
) -> PixelateResult {
    let w_out = params.w_out;
    let h_out = params.h_out;
    let n_out = w_out * h_out;

    // --- Initialization (Section 4.1) ---------------------------------------
    let mean_color = mean(lab);
    let uniform = vec![1.0; lab.len()];
    let tc = critical_temperature(lab, &uniform);
    let t0 = (1.1 * tc).max(params.t_final);
    let (cov, _) = covariance(lab);
    let (axis, _) = principal_axis(&cov);

    let mut sps = init_grid(w_in, h_in, w_out, h_out, mean_color);
    let mut palette = Palette::new(mean_color, axis, params.palette_size, t0, params.anneal);
    let mut sp_palette = vec![0usize; n_out];
    let mut assignment = assign(lab, w_in, h_in, &sps, w_out, h_out, params.m);
    let mut smoothed: Vec<Lab> = vec![mean_color; n_out];

    let mut iterations = 0;
    let mut converged = false;

    // --- Main loop (Algorithm 1) --------------------------------------------
    while iterations < params.max_iter {
        iterations += 1;

        // Superpixel refinement (Section 4.2): colors come from the palette.
        for (s, sp) in sps.iter_mut().enumerate() {
            sp.color = palette.color(sp_palette[s]);
        }
        assignment = assign(lab, w_in, h_in, &sps, w_out, h_out, params.m);
        update(lab, w_in, &assignment, importance, &mut sps);
        laplacian_smooth(&mut sps, w_out, h_out, params.laplacian);

        let ms: Vec<Lab> = sps.iter().map(|s| s.mean_color).collect();
        smoothed = if params.bilateral {
            bilateral(&ms, w_out, h_out, params.bilateral_sigma_s, params.bilateral_sigma_r)
        } else {
            ms
        };

        // Palette refinement (Section 4.3).
        let prior = superpixel_prior(&sps, importance.is_some());
        let assoc = palette.associate(&smoothed, &prior);
        let change = palette.refine(&smoothed, &prior, &assoc);
        sp_palette = palette.assign(&assoc);

        if let Some(cb) = progress.as_deref_mut() {
            cb(iterations, palette.temperature, palette.len());
        }

        if change < params.anneal.eps_palette {
            if palette.temperature <= params.t_final {
                converged = true;
                break;
            }
            palette.expand(&smoothed, &prior, &assoc);
        }
    }

    PixelateResult {
        assignment,
        palette_id: sp_palette,
        palette: palette.colors(),
        superpixels: sps,
        smoothed_colors: smoothed,
        iterations,
        temperature: palette.temperature,
        initial_temperature: t0,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadrant_image(w: usize, h: usize) -> Vec<Lab> {
        let cols = [
            [20.0, 0.0, 0.0],
            [80.0, 0.0, 0.0],
            [50.0, 60.0, 0.0],
            [50.0, 0.0, -60.0],
        ];
        let mut lab = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let q = (if x >= w / 2 { 1 } else { 0 }) + (if y >= h / 2 { 2 } else { 0 });
                lab.push(cols[q]);
            }
        }
        lab
    }

    #[test]
    fn recovers_four_quadrant_colors() {
        let lab = quadrant_image(32, 32);
        let params = PixelateParams {
            w_out: 4,
            h_out: 4,
            palette_size: 4,
            m: 45.0,
            t_final: 1.0,
            laplacian: 0.4,
            bilateral: true,
            bilateral_sigma_s: 1.0,
            bilateral_sigma_r: 10.0,
            max_iter: 2000,
            anneal: AnnealParams::default(),
        };
        let res = pixelate(&lab, 32, 32, None, &params, None);
        assert!(res.converged);
        assert_eq!(res.palette.len(), 4);
        assert_eq!(res.palette_id.len(), 16);
        // Top-left output pixel should be dark, bottom-right blue-ish.
        let tl = res.palette[res.palette_id[0]];
        let br = res.palette[res.palette_id[15]];
        assert!(tl[0] < 30.0);
        assert!(br[2] < -40.0);
    }
}
