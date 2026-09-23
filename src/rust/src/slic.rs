//! Modified SLIC superpixel segmentation (Gerstner et al. 2012, Section 4.2).
//!
//! Each superpixel corresponds to exactly one output pixel. Compared to the
//! original SLIC algorithm (Achanta et al. 2010) three changes are made:
//!
//! * the color used when assigning input pixels is the *palette* color
//!   associated with the superpixel rather than its mean color,
//! * a larger positional weight `m` is used (45 instead of 10),
//! * superpixel centers are Laplacian smoothed towards the average position of
//!   their 4-connected neighbors in the output grid after each update.

use crate::color::Lab;

/// A superpixel: a region of the input image that maps to one output pixel.
#[derive(Clone, Debug)]
pub struct Superpixel {
    /// Center x position in input pixel coordinates.
    pub cx: f64,
    /// Center y position in input pixel coordinates.
    pub cy: f64,
    /// Column of the corresponding output pixel.
    pub grid_x: usize,
    /// Row of the corresponding output pixel.
    pub grid_y: usize,
    /// Representative color used during assignment (the palette color).
    pub color: Lab,
    /// Mean color `m_s` of the assigned input pixels.
    pub mean_color: Lab,
    /// Number of assigned input pixels.
    pub count: usize,
    /// Mean importance of the assigned input pixels (1 when no map is given).
    pub importance: f64,
}

/// Initialize `w_out * h_out` superpixels on a regular grid over the input.
pub fn init_grid(w_in: usize, h_in: usize, w_out: usize, h_out: usize, color: Lab) -> Vec<Superpixel> {
    let sx = w_in as f64 / w_out as f64;
    let sy = h_in as f64 / h_out as f64;
    let mut out = Vec::with_capacity(w_out * h_out);
    for gy in 0..h_out {
        for gx in 0..w_out {
            out.push(Superpixel {
                cx: (gx as f64 + 0.5) * sx,
                cy: (gy as f64 + 0.5) * sy,
                grid_x: gx,
                grid_y: gy,
                color,
                mean_color: color,
                count: 0,
                importance: 1.0,
            });
        }
    }
    out
}

/// Assign every input pixel to the superpixel minimizing
/// `d = d_c + m * sqrt(N / M) * d_p` (Equation 1 of the paper).
///
/// `lab` is row-major (`index = y * w_in + x`). Each superpixel only competes
/// for pixels inside a window around its center, as in SLIC, which keeps the
/// cost proportional to the number of input pixels. Any pixel that falls
/// outside every window (possible only after large center drift) is assigned
/// to the positionally nearest superpixel.
pub fn assign(
    lab: &[Lab],
    w_in: usize,
    h_in: usize,
    sps: &[Superpixel],
    w_out: usize,
    h_out: usize,
    m: f64,
) -> Vec<usize> {
    let n_in = lab.len();
    let n_out = sps.len();
    debug_assert_eq!(n_in, w_in * h_in);
    debug_assert_eq!(n_out, w_out * h_out);

    let pos_weight = m * (n_out as f64 / n_in as f64).sqrt();
    let sx = w_in as f64 / w_out as f64;
    let sy = h_in as f64 / h_out as f64;
    let rx = (2.0 * sx).ceil().max(1.0) as i64;
    let ry = (2.0 * sy).ceil().max(1.0) as i64;

    let mut best = vec![f64::INFINITY; n_in];
    let mut label = vec![usize::MAX; n_in];

    for (s, sp) in sps.iter().enumerate() {
        let cxi = sp.cx.round() as i64;
        let cyi = sp.cy.round() as i64;
        let x0 = (cxi - rx).max(0) as usize;
        let x1 = ((cxi + rx).min(w_in as i64 - 1)).max(0) as usize;
        let y0 = (cyi - ry).max(0) as usize;
        let y1 = ((cyi + ry).min(h_in as i64 - 1)).max(0) as usize;
        for y in y0..=y1 {
            let dy = y as f64 - sp.cy;
            let row = y * w_in;
            for x in x0..=x1 {
                let i = row + x;
                let dx = x as f64 - sp.cx;
                let dp = (dx * dx + dy * dy).sqrt();
                let dc = crate::color::dist(&lab[i], &sp.color);
                let d = dc + pos_weight * dp;
                if d < best[i] {
                    best[i] = d;
                    label[i] = s;
                }
            }
        }
    }

    // Fallback for pixels no window reached.
    for i in 0..n_in {
        if label[i] == usize::MAX {
            let x = (i % w_in) as f64;
            let y = (i / w_in) as f64;
            let mut bd = f64::INFINITY;
            for (s, sp) in sps.iter().enumerate() {
                let dx = x - sp.cx;
                let dy = y - sp.cy;
                let d = dx * dx + dy * dy;
                if d < bd {
                    bd = d;
                    label[i] = s;
                }
            }
        }
    }
    label
}

/// Update superpixel centers, mean colors, counts and importance from an
/// assignment. Superpixels with no assigned pixels keep their previous state.
pub fn update(
    lab: &[Lab],
    w_in: usize,
    assignment: &[usize],
    importance: Option<&[f64]>,
    sps: &mut [Superpixel],
) {
    let n_out = sps.len();
    let mut sum_x = vec![0.0; n_out];
    let mut sum_y = vec![0.0; n_out];
    let mut sum_c = vec![[0.0; 3]; n_out];
    let mut sum_w = vec![0.0; n_out];
    let mut count = vec![0usize; n_out];

    for (i, &s) in assignment.iter().enumerate() {
        let x = (i % w_in) as f64;
        let y = (i / w_in) as f64;
        sum_x[s] += x;
        sum_y[s] += y;
        sum_c[s][0] += lab[i][0];
        sum_c[s][1] += lab[i][1];
        sum_c[s][2] += lab[i][2];
        sum_w[s] += importance.map_or(1.0, |imp| imp[i]);
        count[s] += 1;
    }

    for (s, sp) in sps.iter_mut().enumerate() {
        sp.count = count[s];
        if count[s] == 0 {
            continue;
        }
        let n = count[s] as f64;
        sp.cx = sum_x[s] / n;
        sp.cy = sum_y[s] / n;
        sp.mean_color = [sum_c[s][0] / n, sum_c[s][1] / n, sum_c[s][2] / n];
        sp.importance = sum_w[s] / n;
    }
}

/// Move each superpixel center `weight` of the way towards the average
/// position of its 4-connected neighbors in the output grid.
///
/// Positions are read from a snapshot so the result does not depend on the
/// order superpixels are visited.
pub fn laplacian_smooth(sps: &mut [Superpixel], w_out: usize, h_out: usize, weight: f64) {
    if weight <= 0.0 {
        return;
    }
    let snapshot: Vec<(f64, f64)> = sps.iter().map(|s| (s.cx, s.cy)).collect();
    for sp in sps.iter_mut() {
        let gx = sp.grid_x as i64;
        let gy = sp.grid_y as i64;
        let mut ax = 0.0;
        let mut ay = 0.0;
        let mut n = 0.0;
        for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
            let nx = gx + dx;
            let ny = gy + dy;
            if nx < 0 || ny < 0 || nx >= w_out as i64 || ny >= h_out as i64 {
                continue;
            }
            let idx = ny as usize * w_out + nx as usize;
            ax += snapshot[idx].0;
            ay += snapshot[idx].1;
            n += 1.0;
        }
        if n > 0.0 {
            ax /= n;
            ay /= n;
            sp.cx += weight * (ax - sp.cx);
            sp.cy += weight * (ay - sp.cy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_initialization_is_regular() {
        let sps = init_grid(8, 4, 4, 2, [50.0, 0.0, 0.0]);
        assert_eq!(sps.len(), 8);
        assert_eq!(sps[0].cx, 1.0);
        assert_eq!(sps[0].cy, 1.0);
        assert_eq!(sps[7].cx, 7.0);
        assert_eq!(sps[7].cy, 3.0);
    }

    #[test]
    fn two_tone_image_splits_left_right() {
        // 8 x 4 image: left half dark, right half light; 2 x 1 output.
        let w = 8;
        let h = 4;
        let mut lab = Vec::new();
        for _y in 0..h {
            for x in 0..w {
                lab.push(if x < 4 { [10.0, 0.0, 0.0] } else { [90.0, 0.0, 0.0] });
            }
        }
        let mut sps = init_grid(w, h, 2, 1, [50.0, 0.0, 0.0]);
        sps[0].color = [10.0, 0.0, 0.0];
        sps[1].color = [90.0, 0.0, 0.0];
        let a = assign(&lab, w, h, &sps, 2, 1, 45.0);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(a[y * w + x], if x < 4 { 0 } else { 1 });
            }
        }
    }
}
