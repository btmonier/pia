//! Bilateral filtering of the superpixel color image.
//!
//! The paper (Section 4.2) smooths the mean colors `m_s` of the superpixels
//! by arranging them in a `w_out x h_out` image and applying a bilateral
//! filter. The filtered colors `m_s'` are used when iterating the palette so
//! that smooth gradients in the input stay continuous in the output.

use crate::color::{dist_sq, Lab};

/// Apply a bilateral filter to a row-major `w x h` grid of colors.
///
/// `sigma_s` is the spatial standard deviation in output pixels and
/// `sigma_r` the range standard deviation in CIELAB units. A non-positive
/// `sigma_s` returns the input unchanged.
pub fn bilateral(colors: &[Lab], w: usize, h: usize, sigma_s: f64, sigma_r: f64) -> Vec<Lab> {
    debug_assert_eq!(colors.len(), w * h);
    if sigma_s <= 0.0 || colors.is_empty() {
        return colors.to_vec();
    }
    let radius = (2.0 * sigma_s).ceil().max(1.0) as i64;
    let inv_s = 1.0 / (2.0 * sigma_s * sigma_s);
    let inv_r = if sigma_r > 0.0 { 1.0 / (2.0 * sigma_r * sigma_r) } else { 0.0 };

    let mut out = vec![[0.0; 3]; colors.len()];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let i = (y as usize) * w + x as usize;
            let center = colors[i];
            let mut acc = [0.0; 3];
            let mut total = 0.0;
            for dy in -radius..=radius {
                let ny = y + dy;
                if ny < 0 || ny >= h as i64 {
                    continue;
                }
                for dx in -radius..=radius {
                    let nx = x + dx;
                    if nx < 0 || nx >= w as i64 {
                        continue;
                    }
                    let j = (ny as usize) * w + nx as usize;
                    let ds = (dx * dx + dy * dy) as f64;
                    let dr = dist_sq(&colors[j], &center);
                    let wgt = (-(ds * inv_s) - dr * inv_r).exp();
                    acc[0] += wgt * colors[j][0];
                    acc[1] += wgt * colors[j][1];
                    acc[2] += wgt * colors[j][2];
                    total += wgt;
                }
            }
            out[i] = if total > 0.0 {
                [acc[0] / total, acc[1] / total, acc[2] / total]
            } else {
                center
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_image_is_unchanged() {
        let colors = vec![[50.0, 10.0, -10.0]; 12];
        let out = bilateral(&colors, 4, 3, 1.0, 10.0);
        for c in out {
            assert!((c[0] - 50.0).abs() < 1e-9);
            assert!((c[1] - 10.0).abs() < 1e-9);
            assert!((c[2] + 10.0).abs() < 1e-9);
        }
    }

    #[test]
    fn strong_edge_is_preserved() {
        let mut colors = Vec::new();
        for _y in 0..4 {
            for x in 0..4 {
                colors.push(if x < 2 { [0.0, 0.0, 0.0] } else { [100.0, 0.0, 0.0] });
            }
        }
        let out = bilateral(&colors, 4, 4, 1.0, 5.0);
        assert!(out[0][0] < 1.0);
        assert!(out[3][0] > 99.0);
    }
}
