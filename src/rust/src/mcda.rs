//! Palette optimization by mass-constrained deterministic annealing
//! (Rose 1998), adapted as described in Section 4.3 of Gerstner et al. 2012.
//!
//! Every palette color `c_k` is represented by two sub-clusters. The
//! association and refinement steps treat each sub-cluster as a separate
//! cluster; the color of `c_k` is the mean of its sub-clusters. Once the
//! palette has converged at the current temperature the temperature is
//! lowered and colors whose sub-clusters have drifted apart are split. When
//! the maximum palette size is reached sub-clusters are collapsed into single
//! clusters.

use crate::color::{dist, principal_axis, weighted_covariance, Lab};

/// Lower bound for cluster probabilities, avoiding clusters that can never
/// recover once their mass hits exactly zero through underflow.
const MIN_PROB: f64 = 1e-12;

/// A single cluster point with its marginal probability `P(c)`.
#[derive(Clone, Debug)]
pub struct Cluster {
    pub color: Lab,
    pub prob: f64,
}

/// A palette color composed of one or two sub-clusters.
#[derive(Clone, Debug)]
pub struct PaletteColor {
    pub clusters: Vec<Cluster>,
}

impl PaletteColor {
    /// Mean color of the sub-clusters.
    pub fn color(&self) -> Lab {
        let n = self.clusters.len() as f64;
        let mut acc = [0.0; 3];
        for c in &self.clusters {
            acc[0] += c.color[0];
            acc[1] += c.color[1];
            acc[2] += c.color[2];
        }
        [acc[0] / n, acc[1] / n, acc[2] / n]
    }

    /// Total probability mass of the color.
    pub fn prob(&self) -> f64 {
        self.clusters.iter().map(|c| c.prob).sum()
    }

    /// Distance between the two sub-clusters (zero for a single cluster).
    pub fn separation(&self) -> f64 {
        if self.clusters.len() == 2 {
            dist(&self.clusters[0].color, &self.clusters[1].color)
        } else {
            0.0
        }
    }
}

/// Tunable constants of the annealing schedule.
#[derive(Clone, Copy, Debug)]
pub struct AnnealParams {
    /// Temperature reduction factor `alpha` applied on convergence.
    pub alpha: f64,
    /// Mean per-cluster palette change below which the palette is
    /// considered converged.
    pub eps_palette: f64,
    /// Sub-cluster separation above which a color is split.
    pub eps_cluster: f64,
    /// Magnitude of the sub-cluster perturbation along the principal axis.
    pub perturbation: f64,
}

impl Default for AnnealParams {
    fn default() -> Self {
        AnnealParams {
            alpha: 0.7,
            eps_palette: 0.25,
            eps_cluster: 1.0,
            perturbation: 0.1,
        }
    }
}

/// Conditional probabilities `P(c_j | p_s)` for every cluster `j` and
/// superpixel `s`, stored row-major with one row per superpixel.
pub struct Association {
    pub n_superpixels: usize,
    pub n_clusters: usize,
    pub probs: Vec<f64>,
    /// Palette color owning each cluster column.
    pub owner: Vec<usize>,
}

impl Association {
    #[inline]
    pub fn get(&self, s: usize, j: usize) -> f64 {
        self.probs[s * self.n_clusters + j]
    }

    /// `P(c_k | p_s)` for palette color `k`, summing over its sub-clusters.
    pub fn palette_prob(&self, s: usize, k: usize) -> f64 {
        let mut p = 0.0;
        for j in 0..self.n_clusters {
            if self.owner[j] == k {
                p += self.get(s, j);
            }
        }
        p
    }
}

/// The evolving palette together with the annealing temperature.
#[derive(Clone, Debug)]
pub struct Palette {
    pub colors: Vec<PaletteColor>,
    pub max_colors: usize,
    pub temperature: f64,
    pub params: AnnealParams,
}

impl Palette {
    /// Create a palette with a single color whose sub-clusters are perturbed
    /// along `axis`.
    pub fn new(initial: Lab, axis: Lab, max_colors: usize, temperature: f64, params: AnnealParams) -> Self {
        let max_colors = max_colors.max(1);
        let mut palette = Palette {
            colors: vec![PaletteColor {
                clusters: vec![Cluster { color: initial, prob: 1.0 }],
            }],
            max_colors,
            temperature,
            params,
        };
        if max_colors > 1 {
            palette.colors[0].clusters = split_clusters(initial, axis, 1.0, params.perturbation);
        }
        palette
    }

    /// Number of distinct palette colors.
    pub fn len(&self) -> usize {
        self.colors.len()
    }

    /// Color of palette entry `k`.
    pub fn color(&self, k: usize) -> Lab {
        self.colors[k].color()
    }

    /// All palette colors.
    pub fn colors(&self) -> Vec<Lab> {
        self.colors.iter().map(|c| c.color()).collect()
    }

    fn cluster_view(&self) -> (Vec<Lab>, Vec<f64>, Vec<usize>) {
        let mut colors = Vec::new();
        let mut probs = Vec::new();
        let mut owner = Vec::new();
        for (k, pc) in self.colors.iter().enumerate() {
            for c in &pc.clusters {
                colors.push(c.color);
                probs.push(c.prob);
                owner.push(k);
            }
        }
        (colors, probs, owner)
    }

    fn set_cluster_probs(&mut self, probs: &[f64]) {
        let mut j = 0;
        for pc in self.colors.iter_mut() {
            for c in pc.clusters.iter_mut() {
                c.prob = probs[j].max(MIN_PROB);
                j += 1;
            }
        }
    }

    fn set_cluster_colors(&mut self, colors: &[Lab]) {
        let mut j = 0;
        for pc in self.colors.iter_mut() {
            for c in pc.clusters.iter_mut() {
                c.color = colors[j];
                j += 1;
            }
        }
    }

    /// Associate step (Equations 2 and 3): compute `P(c_j | p_s)` for every
    /// cluster and superpixel, then update the marginals `P(c_j)`.
    ///
    /// `prior` holds `P(p_s)` and must sum to one.
    pub fn associate(&mut self, ms: &[Lab], prior: &[f64]) -> Association {
        let (colors, probs, owner) = self.cluster_view();
        let assoc = conditional_probabilities(ms, &colors, &probs, self.temperature, &owner);

        let mut marginal = vec![0.0; colors.len()];
        for s in 0..ms.len() {
            for j in 0..colors.len() {
                marginal[j] += assoc.get(s, j) * prior[s];
            }
        }
        self.set_cluster_probs(&marginal);
        assoc
    }

    /// Refine step (Equation 4): move every cluster to the probability
    /// weighted mean of the superpixel colors. Returns the mean movement
    /// per cluster, which is compared against `eps_palette` for convergence.
    pub fn refine(&mut self, ms: &[Lab], prior: &[f64], assoc: &Association) -> f64 {
        let (old_colors, probs, _) = self.cluster_view();
        let mut new_colors = old_colors.clone();
        for j in 0..old_colors.len() {
            let mut acc = [0.0; 3];
            let mut total = 0.0;
            for s in 0..ms.len() {
                let w = assoc.get(s, j) * prior[s];
                acc[0] += w * ms[s][0];
                acc[1] += w * ms[s][1];
                acc[2] += w * ms[s][2];
                total += w;
            }
            // `total` equals the freshly computed marginal P(c_j); guard tiny masses.
            if total > MIN_PROB && probs[j] > MIN_PROB {
                new_colors[j] = [acc[0] / total, acc[1] / total, acc[2] / total];
            }
        }
        let change: f64 = old_colors
            .iter()
            .zip(&new_colors)
            .map(|(a, b)| dist(a, b))
            .sum::<f64>()
            / old_colors.len().max(1) as f64;
        self.set_cluster_colors(&new_colors);
        change
    }

    /// Assign each superpixel to the palette color maximizing `P(c_k | p_s)`.
    pub fn assign(&self, assoc: &Association) -> Vec<usize> {
        let k = self.len();
        (0..assoc.n_superpixels)
            .map(|s| {
                let mut best = 0;
                let mut bp = -1.0;
                for c in 0..k {
                    let p = assoc.palette_prob(s, c);
                    if p > bp {
                        bp = p;
                        best = c;
                    }
                }
                best
            })
            .collect()
    }

    /// Expand step: lower the temperature, split colors whose sub-clusters
    /// have separated, then re-perturb sub-clusters (or collapse them once
    /// the maximum palette size is reached). Returns the number of splits.
    pub fn expand(&mut self, ms: &[Lab], prior: &[f64], assoc: &Association) -> usize {
        self.temperature *= self.params.alpha;

        let mut n_split = 0;
        if self.len() < self.max_colors {
            let mut candidates: Vec<(usize, f64)> = self
                .colors
                .iter()
                .enumerate()
                .map(|(k, c)| (k, c.separation()))
                .filter(|(_, sep)| *sep > self.params.eps_cluster)
                .collect();
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (k, _) in candidates {
                if self.len() >= self.max_colors {
                    break;
                }
                let second = self.colors[k].clusters.remove(1);
                self.colors.push(PaletteColor { clusters: vec![second] });
                n_split += 1;
            }
        }

        if self.len() >= self.max_colors {
            // Collapse sub-clusters into a single cluster per color.
            for pc in self.colors.iter_mut() {
                let color = pc.color();
                let prob = pc.prob();
                pc.clusters = vec![Cluster { color, prob }];
            }
            return n_split;
        }

        // Re-perturb colors along the principal axis of the superpixel
        // colors they are responsible for. Colors whose sub-clusters are
        // already drifting apart (but have not yet reached the split
        // threshold) are left alone so that progress made at the previous
        // temperature is not thrown away.
        let weights_for = |k: usize, assoc: &Association| -> Vec<f64> {
            (0..ms.len()).map(|s| assoc.palette_prob(s, k) * prior[s]).collect()
        };
        let n_old = assoc.owner.iter().copied().max().map_or(0, |m| m + 1);
        for k in 0..self.len() {
            if self.colors[k].separation() > 2.0 * self.params.perturbation {
                continue;
            }
            let color = self.colors[k].color();
            let prob = self.colors[k].prob();
            // Newly split colors have no column in `assoc`; use the parent's
            // responsibilities, which is a reasonable proxy for the axis.
            let axis = if k < n_old {
                let w = weights_for(k, assoc);
                let (cov, _) = weighted_covariance(ms, &w);
                principal_axis(&cov).0
            } else {
                [1.0, 0.0, 0.0]
            };
            self.colors[k].clusters = split_clusters(color, axis, prob, self.params.perturbation);
        }
        n_split
    }
}

/// Two sub-clusters displaced `perturbation` along `axis` either side of
/// `color`, sharing `prob` equally.
fn split_clusters(color: Lab, axis: Lab, prob: f64, perturbation: f64) -> Vec<Cluster> {
    let p = prob / 2.0;
    vec![
        Cluster {
            color: [
                color[0] + perturbation * axis[0],
                color[1] + perturbation * axis[1],
                color[2] + perturbation * axis[2],
            ],
            prob: p,
        },
        Cluster {
            color: [
                color[0] - perturbation * axis[0],
                color[1] - perturbation * axis[1],
                color[2] - perturbation * axis[2],
            ],
            prob: p,
        },
    ]
}

/// Compute `P(c_j | p_s) ∝ P(c_j) exp(-||m_s - c_j|| / T)` for all `s`, `j`,
/// normalized over `j`. Uses a log-sum-exp so very low temperatures do not
/// underflow to all-zero rows.
pub fn conditional_probabilities(
    ms: &[Lab],
    cluster_colors: &[Lab],
    cluster_probs: &[f64],
    temperature: f64,
    owner: &[usize],
) -> Association {
    let n = ms.len();
    let j_n = cluster_colors.len();
    let t = temperature.max(f64::MIN_POSITIVE);
    let log_prior: Vec<f64> = cluster_probs
        .iter()
        .map(|p| if *p > 0.0 { p.ln() } else { f64::NEG_INFINITY })
        .collect();
    let all_dead = log_prior.iter().all(|v| !v.is_finite());

    let mut probs = vec![0.0; n * j_n];
    let mut logw = vec![0.0; j_n];
    for s in 0..n {
        let mut max = f64::NEG_INFINITY;
        for j in 0..j_n {
            let lp = if all_dead { 0.0 } else { log_prior[j] };
            let v = lp - dist(&ms[s], &cluster_colors[j]) / t;
            logw[j] = v;
            if v > max {
                max = v;
            }
        }
        let mut total = 0.0;
        for j in 0..j_n {
            let w = (logw[j] - max).exp();
            probs[s * j_n + j] = w;
            total += w;
        }
        for j in 0..j_n {
            probs[s * j_n + j] /= total;
        }
    }
    Association {
        n_superpixels: n,
        n_clusters: j_n,
        probs,
        owner: owner.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditional_probabilities_sum_to_one() {
        let ms = vec![[10.0, 0.0, 0.0], [90.0, 0.0, 0.0]];
        let cols = vec![[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]];
        let a = conditional_probabilities(&ms, &cols, &[0.5, 0.5], 5.0, &[0, 1]);
        for s in 0..2 {
            let sum: f64 = (0..2).map(|j| a.get(s, j)).sum();
            assert!((sum - 1.0).abs() < 1e-12);
        }
        assert!(a.get(0, 0) > 0.99);
        assert!(a.get(1, 1) > 0.99);
    }

    #[test]
    fn high_temperature_is_uniform() {
        let ms = vec![[10.0, 0.0, 0.0]];
        let cols = vec![[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]];
        let a = conditional_probabilities(&ms, &cols, &[0.5, 0.5], 1e9, &[0, 1]);
        assert!((a.get(0, 0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn palette_splits_two_tone_input() {
        let ms: Vec<Lab> = (0..20)
            .map(|i| if i < 10 { [10.0, 0.0, 0.0] } else { [90.0, 0.0, 0.0] })
            .collect();
        let prior = vec![1.0 / 20.0; 20];
        let mut pal = Palette::new([50.0, 0.0, 0.0], [1.0, 0.0, 0.0], 2, 4000.0, AnnealParams::default());
        let mut iters = 0;
        loop {
            iters += 1;
            let a = pal.associate(&ms, &prior);
            let change = pal.refine(&ms, &prior, &a);
            if change < pal.params.eps_palette {
                if pal.temperature <= 1.0 {
                    break;
                }
                pal.expand(&ms, &prior, &a);
            }
            assert!(iters < 5000);
        }
        assert_eq!(pal.len(), 2);
        let mut cols = pal.colors();
        cols.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        assert!((cols[0][0] - 10.0).abs() < 1.0);
        assert!((cols[1][0] - 90.0).abs() < 1.0);
    }
}
