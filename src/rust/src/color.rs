//! Helpers for working with colors in CIELAB space.
//!
//! All colors are represented as `[L, a, b]` triples. The routines here are
//! deliberately small and allocation free so they can be used in the hot loops
//! of the SLIC and deterministic annealing steps.

/// A color in CIELAB space stored as `[L, a, b]`.
pub type Lab = [f64; 3];

/// A symmetric 3 x 3 covariance matrix.
pub type Cov3 = [[f64; 3]; 3];

/// Squared Euclidean distance between two colors.
#[inline]
pub fn dist_sq(a: &Lab, b: &Lab) -> f64 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2
}

/// Euclidean distance between two colors.
#[inline]
pub fn dist(a: &Lab, b: &Lab) -> f64 {
    dist_sq(a, b).sqrt()
}

/// Unweighted mean of a set of colors. Returns black for an empty set.
pub fn mean(colors: &[Lab]) -> Lab {
    if colors.is_empty() {
        return [0.0; 3];
    }
    let mut acc = [0.0; 3];
    for c in colors {
        acc[0] += c[0];
        acc[1] += c[1];
        acc[2] += c[2];
    }
    let n = colors.len() as f64;
    [acc[0] / n, acc[1] / n, acc[2] / n]
}

/// Weighted mean of a set of colors. Returns `None` when the total weight
/// is not positive.
pub fn weighted_mean(colors: &[Lab], weights: &[f64]) -> Option<Lab> {
    debug_assert_eq!(colors.len(), weights.len());
    let mut acc = [0.0; 3];
    let mut total = 0.0;
    for (c, &w) in colors.iter().zip(weights) {
        acc[0] += w * c[0];
        acc[1] += w * c[1];
        acc[2] += w * c[2];
        total += w;
    }
    if total <= 0.0 {
        None
    } else {
        Some([acc[0] / total, acc[1] / total, acc[2] / total])
    }
}

/// Weighted covariance matrix of a set of colors around their weighted mean.
///
/// Weights are normalized internally so they need not sum to one. Returns the
/// covariance together with the weighted mean. If the total weight is not
/// positive a zero matrix and black mean are returned.
pub fn weighted_covariance(colors: &[Lab], weights: &[f64]) -> (Cov3, Lab) {
    let Some(mu) = weighted_mean(colors, weights) else {
        return ([[0.0; 3]; 3], [0.0; 3]);
    };
    let total: f64 = weights.iter().sum();
    let mut cov = [[0.0; 3]; 3];
    for (c, &w) in colors.iter().zip(weights) {
        if w <= 0.0 {
            continue;
        }
        let d = [c[0] - mu[0], c[1] - mu[1], c[2] - mu[2]];
        for i in 0..3 {
            for j in i..3 {
                cov[i][j] += w * d[i] * d[j];
            }
        }
    }
    for i in 0..3 {
        for j in i..3 {
            cov[i][j] /= total;
            cov[j][i] = cov[i][j];
        }
    }
    (cov, mu)
}

/// Unweighted covariance of a set of colors.
pub fn covariance(colors: &[Lab]) -> (Cov3, Lab) {
    let w = vec![1.0; colors.len()];
    weighted_covariance(colors, &w)
}

fn normalize(v: &mut Lab) -> f64 {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 0.0 {
        v[0] /= n;
        v[1] /= n;
        v[2] /= n;
    }
    n
}

fn mat_vec(m: &Cov3, v: &Lab) -> Lab {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Principal axis (unit eigenvector with the largest eigenvalue) of a
/// symmetric positive semi-definite matrix, together with that eigenvalue.
///
/// Uses power iteration, which is deterministic and converges quickly for
/// the well conditioned 3 x 3 matrices that arise here. A zero matrix yields
/// the `L` axis with zero variance.
pub fn principal_axis(cov: &Cov3) -> (Lab, f64) {
    let trace = cov[0][0] + cov[1][1] + cov[2][2];
    if !(trace > 0.0) {
        return ([1.0, 0.0, 0.0], 0.0);
    }
    // A start vector with distinct irrational-ish components is extremely
    // unlikely to be orthogonal to the dominant eigenvector.
    let mut v: Lab = [0.7071, 0.5774, 0.4082];
    normalize(&mut v);
    let mut lambda = 0.0;
    for _ in 0..256 {
        let mut next = mat_vec(cov, &v);
        let n = normalize(&mut next);
        if n == 0.0 {
            break;
        }
        let delta = dist_sq(&next, &v);
        v = next;
        lambda = n;
        if delta < 1e-24 {
            break;
        }
    }
    // Rayleigh quotient gives the eigenvalue for the converged direction.
    let cv = mat_vec(cov, &v);
    let rq = v[0] * cv[0] + v[1] * cv[1] + v[2] * cv[2];
    if rq.is_finite() {
        lambda = rq;
    }
    (v, lambda.max(0.0))
}

/// Critical temperature of a weighted set of colors, defined by Rose (1998)
/// as twice the variance along the major principal component axis.
pub fn critical_temperature(colors: &[Lab], weights: &[f64]) -> f64 {
    let (cov, _) = weighted_covariance(colors, weights);
    let (_, var) = principal_axis(&cov);
    2.0 * var
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn principal_axis_recovers_dominant_direction() {
        let colors: Vec<Lab> = (0..100)
            .map(|i| {
                let t = i as f64;
                [t, 0.1 * t, 0.0]
            })
            .collect();
        let (cov, _) = covariance(&colors);
        let (axis, var) = principal_axis(&cov);
        assert!((axis[0].abs() - 0.995).abs() < 0.01);
        assert!(var > 800.0);
    }

    #[test]
    fn critical_temperature_is_twice_variance() {
        let colors: Vec<Lab> = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let tc = critical_temperature(&colors, &[1.0, 1.0]);
        // variance along L is 25, so Tc = 50
        assert!((tc - 50.0).abs() < 1e-9);
    }
}
