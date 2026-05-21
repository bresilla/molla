//! Stage F1 — CPU oracle for the WCSPH GPU kernels.
//!
//! Pure `f64`, `wgpu`-free brute-force (`O(N²)`) SPH stepper that
//! reproduces the exact math of `shaders/sph.wgsl`. It exists so the
//! default `cargo test` lane can verify the WCSPH formulas, and so
//! `tests/sph_gpu.rs` can check CPU/GPU parity. It is a reference, not
//! a performant simulator — there is no neighbour grid.

/// Cubic-spline kernel (Monaghan), 3D normalisation. Support is `q < 2`.
pub fn kernel_w(r: f64, h: f64) -> f64 {
    let q = r / h;
    let sigma = 1.0 / (std::f64::consts::PI * h * h * h);
    if q < 1.0 {
        sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    } else if q < 2.0 {
        let t = 2.0 - q;
        sigma * 0.25 * t * t * t
    } else {
        0.0
    }
}

/// `∇_i W(r_i - r_j)` for separation vector `rvec` of length `r`.
pub fn kernel_grad_w(rvec: [f64; 3], r: f64, h: f64) -> [f64; 3] {
    if r < 1e-7 {
        return [0.0; 3];
    }
    let q = r / h;
    let sigma = 1.0 / (std::f64::consts::PI * h * h * h);
    let dwdq = if q < 1.0 {
        sigma * (-3.0 * q + 2.25 * q * q)
    } else if q < 2.0 {
        let t = 2.0 - q;
        sigma * (-0.75 * t * t)
    } else {
        0.0
    };
    let s = (dwdq / h) / r;
    [s * rvec[0], s * rvec[1], s * rvec[2]]
}

/// Tait equation of state, clamped to non-negative pressure.
pub fn tait_pressure(density: f64, rest_density: f64, b: f64, gamma: f64) -> f64 {
    (b * ((density / rest_density).powf(gamma) - 1.0)).max(0.0)
}

/// F4 — granular regularisation safety factor (mirrors `sph.wgsl`).
const STRESS_STAB_K: f64 = 16.0;

/// Brute-force `f64` WCSPH stepper mirroring `shaders/sph.wgsl`.
#[derive(Clone, Debug)]
pub struct SphOracle {
    /// Particle positions — fluid `[0, n_fluid)` then static boundary.
    pub pos: Vec<[f64; 3]>,
    /// Particle velocities.
    pub vel: Vec<[f64; 3]>,
    /// Number of fluid (moving) particles; particles at higher indices
    /// are static boundary particles.
    pub n_fluid: usize,
    /// Per-particle density (filled by [`Self::compute_density`]).
    pub density: Vec<f64>,
    /// Per-particle pressure.
    pub pressure: Vec<f64>,
    /// F4 — per-particle deviatoric stress `[xx,yy,zz,xy,xz,yz]`,
    /// recomputed each step by [`Self::update_stress`].
    pub stress: Vec<[f64; 6]>,
    /// Smoothing length.
    pub h: f64,
    /// Per-particle mass.
    pub mass: f64,
    /// Rest density.
    pub rest_density: f64,
    /// Tait stiffness `B`.
    pub stiffness_b: f64,
    /// Tait exponent.
    pub gamma: f64,
    /// Monaghan artificial-viscosity coefficient.
    pub viscosity: f64,
    /// Gravity vector.
    pub gravity: [f64; 3],
    /// Domain box minimum corner.
    pub domain_min: [f64; 3],
    /// Domain box maximum corner.
    pub domain_max: [f64; 3],
    /// Wall bounce coefficient.
    pub wall_restitution: f64,
    /// Granular Coulomb friction coefficient (F4). `0` = ordinary fluid.
    pub friction: f64,
}

impl SphOracle {
    /// Number of particles.
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    /// Whether the particle set is empty.
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// Sound speed `sqrt(B * gamma / rho0)`.
    pub fn sound_speed(&self) -> f64 {
        (self.stiffness_b * self.gamma / self.rest_density).sqrt()
    }

    /// Recompute density + pressure for all particles (`O(N²)`).
    pub fn compute_density(&mut self) {
        let n = self.len();
        self.density = vec![0.0; n];
        self.pressure = vec![0.0; n];
        for i in 0..n {
            let mut rho = 0.0;
            for j in 0..n {
                let r = dist(self.pos[i], self.pos[j]);
                rho += self.mass * kernel_w(r, self.h);
            }
            rho = rho.max(1e-6);
            self.density[i] = rho;
            self.pressure[i] = tait_pressure(rho, self.rest_density, self.stiffness_b, self.gamma);
        }
    }

    /// Per-particle acceleration. Fluid particles get gravity plus all
    /// neighbours; boundary particles accumulate only fluid neighbours
    /// (the fluid load FSI sums over a body). Mirrors `compute_forces`.
    /// F4 — compute the per-particle deviatoric stress for the granular
    /// rheology. Regularised perfectly-plastic Drucker-Prager:
    /// `S = τ_y · D'/(‖D'‖ + reg)`, `τ_y = friction·pressure`, with a
    /// dt-scaled `reg` pinning the sub-yield viscosity to the stable
    /// ceiling. Mirrors the WGSL `compute_stress` kernel. `S` is aligned
    /// with the strain rate, so `S:D' ≥ 0` — dissipative by construction.
    pub fn update_stress(&mut self, dt: f64) {
        let n = self.len();
        if self.stress.len() != n {
            self.stress = vec![[0.0; 6]; n];
        }
        if self.friction <= 0.0 {
            for s in &mut self.stress {
                *s = [0.0; 6];
            }
            return;
        }
        for i in 0..self.n_fluid {
            let xi = self.pos[i];
            let vi = self.vel[i];
            // Velocity gradient l[a][b] = ∂v_a/∂x_b, plus the SPH-smoothed
            // (Shepard) pressure p_y = Σ V_j p_j W / Σ V_j W used to set
            // the yield — the raw per-particle Tait pressure is too noisy
            // for τ_y.
            let mut l = [[0.0_f64; 3]; 3];
            let voli = self.mass / self.density[i].max(1e-6);
            let w0 = kernel_w(0.0, self.h);
            let mut psum = voli * self.pressure[i].max(0.0) * w0;
            let mut wsum = voli * w0;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let rvec = sub(xi, self.pos[j]);
                let r = norm(rvec);
                if r >= 2.0 * self.h || r < 1e-7 {
                    continue;
                }
                let gw = kernel_grad_w(rvec, r, self.h);
                let dv = sub(self.vel[j], vi);
                let vol = self.mass / self.density[j].max(1e-6);
                for a in 0..3 {
                    for b in 0..3 {
                        l[a][b] += vol * dv[a] * gw[b];
                    }
                }
                let wj = kernel_w(r, self.h);
                psum += vol * self.pressure[j].max(0.0) * wj;
                wsum += vol * wj;
            }
            let dxx = l[0][0];
            let dyy = l[1][1];
            let dzz = l[2][2];
            let dxy = 0.5 * (l[0][1] + l[1][0]);
            let dxz = 0.5 * (l[0][2] + l[2][0]);
            let dyz = 0.5 * (l[1][2] + l[2][1]);
            let tr = (dxx + dyy + dzz) / 3.0;
            // Deviatoric strain rate D' = dev(sym(L)).
            let dprime = [dxx - tr, dyy - tr, dzz - tr, dxy, dxz, dyz];
            let dnorm = (dprime[0] * dprime[0]
                + dprime[1] * dprime[1]
                + dprime[2] * dprime[2]
                + 2.0 * (dprime[3] * dprime[3] + dprime[4] * dprime[4] + dprime[5] * dprime[5]))
                .sqrt();
            // Drucker-Prager yield (from the smoothed pressure) and the
            // dt-scaled regularisation.
            let p_yield = psum / wsum.max(1e-6);
            let tau_y = self.friction * p_yield.max(0.0);
            let reg = (tau_y * STRESS_STAB_K * dt / (2.0 * self.rest_density * self.h * self.h))
                .max(1.0e-4);
            // S = τ_y · D'/(‖D'‖+reg): magnitude ≤ τ_y, aligned with D'.
            let scale = tau_y / (dnorm + reg);
            let mut s = [0.0_f64; 6];
            for k in 0..6 {
                s[k] = scale * dprime[k];
            }
            self.stress[i] = s;
        }
    }

    pub fn compute_accels(&self) -> Vec<[f64; 3]> {
        let n = self.len();
        let c = self.sound_speed();
        let mut accel = vec![[0.0; 3]; n];
        for i in 0..n {
            let i_is_fluid = i < self.n_fluid;
            let mut a = if i_is_fluid { self.gravity } else { [0.0; 3] };
            let xi = self.pos[i];
            let vi = self.vel[i];
            let rho_i = self.density[i];
            let p_i = self.pressure[i];
            for j in 0..n {
                if j == i {
                    continue;
                }
                if !i_is_fluid && j >= self.n_fluid {
                    continue;
                }
                let rvec = sub(xi, self.pos[j]);
                let r = norm(rvec);
                if r >= 2.0 * self.h || r < 1e-7 {
                    continue;
                }
                let rho_j = self.density[j];
                let p_j = self.pressure[j];
                let gw = kernel_grad_w(rvec, r, self.h);
                // Boundary particles mirror the fluid neighbour's
                // pressure (Akinci-style) so a coupled body feels the
                // true fluid pressure field — recovers buoyancy.
                let pterm = if i_is_fluid {
                    p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j)
                } else {
                    2.0 * p_j / (rho_j * rho_j)
                };
                for k in 0..3 {
                    a[k] -= self.mass * pterm * gw[k];
                }
                let vij = sub(vi, self.vel[j]);
                let vr = dot(vij, rvec);
                if vr < 0.0 {
                    let mu = self.h * vr / (dot(rvec, rvec) + 0.01 * self.h * self.h);
                    let rho_bar = 0.5 * (rho_i + rho_j);
                    let visc = -self.viscosity * c * mu / rho_bar;
                    for k in 0..3 {
                        a[k] -= self.mass * visc * gw[k];
                    }
                }
                // F4 — granular deviatoric stress force (mirrors sph.wgsl):
                // a_i += Σ_j m_j (S_i/ρ_i² + S_j/ρ_j²)·∇W.
                if i_is_fluid && self.friction > 0.0 {
                    let wi = 1.0 / (rho_i * rho_i);
                    let wj = 1.0 / (rho_j * rho_j);
                    let si = self.stress[i];
                    let sj = self.stress[j];
                    let mxx = si[0] * wi + sj[0] * wj;
                    let myy = si[1] * wi + sj[1] * wj;
                    let mzz = si[2] * wi + sj[2] * wj;
                    let mxy = si[3] * wi + sj[3] * wj;
                    let mxz = si[4] * wi + sj[4] * wj;
                    let myz = si[5] * wi + sj[5] * wj;
                    a[0] += self.mass * (mxx * gw[0] + mxy * gw[1] + mxz * gw[2]);
                    a[1] += self.mass * (mxy * gw[0] + myy * gw[1] + myz * gw[2]);
                    a[2] += self.mass * (mxz * gw[0] + myz * gw[1] + mzz * gw[2]);
                }
            }
            accel[i] = a;
        }
        accel
    }

    /// Integrate the fluid particles (`[0, n_fluid)`) by one step,
    /// applying the domain-box clamp. The floor additionally applies
    /// Coulomb basal friction for the granular preset — mirrors the
    /// `integrate` kernel in `sph.wgsl`.
    pub fn integrate_fluid(&mut self, accel: &[[f64; 3]], dt: f64) {
        let e = self.wall_restitution;
        for i in 0..self.n_fluid {
            let mut v = [0.0_f64; 3];
            let mut x = [0.0_f64; 3];
            for k in 0..3 {
                v[k] = self.vel[i][k] + accel[i][k] * dt;
                x[k] = self.pos[i][k] + v[k] * dt;
            }
            if x[0] < self.domain_min[0] {
                x[0] = self.domain_min[0];
                v[0] = -e * v[0];
            }
            if x[0] > self.domain_max[0] {
                x[0] = self.domain_max[0];
                v[0] = -e * v[0];
            }
            // Floor: reflect the normal velocity, then apply Coulomb
            // basal friction so a granular pile cannot slide off the
            // floor as a rigid block. The tangential cut is a bounded
            // [0,1] factor → unconditionally stable.
            if x[1] < self.domain_min[1] {
                let vy_in = v[1];
                x[1] = self.domain_min[1];
                v[1] = -e * vy_in;
                if self.friction > 0.0 {
                    let dvn = (vy_in.abs() * (1.0 + e)).max(self.gravity[1].abs() * dt);
                    let vt = (v[0] * v[0] + v[2] * v[2]).sqrt();
                    if vt > 1.0e-8 {
                        let factor = (1.0 - self.friction * dvn / vt).max(0.0);
                        v[0] *= factor;
                        v[2] *= factor;
                    }
                }
            }
            if x[1] > self.domain_max[1] {
                x[1] = self.domain_max[1];
                v[1] = -e * v[1];
            }
            if x[2] < self.domain_min[2] {
                x[2] = self.domain_min[2];
                v[2] = -e * v[2];
            }
            if x[2] > self.domain_max[2] {
                x[2] = self.domain_max[2];
                v[2] = -e * v[2];
            }
            self.vel[i] = v;
            self.pos[i] = x;
        }
    }

    /// Advance one substep (density → forces → integrate the fluid).
    pub fn step(&mut self, dt: f64) {
        if self.len() == 0 {
            return;
        }
        self.compute_density();
        self.update_stress(dt);
        let accel = self.compute_accels();
        self.integrate_fluid(&accel, dt);
    }
}

/// A linear rigid body coupled to [`SphFsiOracle`].
#[derive(Clone, Debug)]
pub struct SphOracleBody {
    /// Body mass.
    pub mass: f64,
    /// Centre-of-mass position.
    pub com: [f64; 3],
    /// Linear velocity.
    pub vel: [f64; 3],
    /// Sample-particle offsets relative to `com`.
    pub offsets: Vec<[f64; 3]>,
    /// Index of this body's first sample particle in `sph.pos`.
    pub start: usize,
}

/// CPU FSI oracle — an [`SphOracle`] plus linear rigid bodies. Mirrors
/// `SphSimulation::step_fsi`.
#[derive(Clone, Debug)]
pub struct SphFsiOracle {
    /// Underlying particle state (fluid then boundary; boundary holds
    /// the body sample particles).
    pub sph: SphOracle,
    /// Coupled rigid bodies.
    pub bodies: Vec<SphOracleBody>,
}

impl SphFsiOracle {
    /// Advance: write the body sample state into the particle arrays,
    /// run one SPH step, then integrate the bodies from their summed
    /// fluid load.
    pub fn step_fsi(&mut self, dt: f64) {
        for b in &self.bodies {
            for (k, off) in b.offsets.iter().enumerate() {
                let idx = b.start + k;
                self.sph.pos[idx] = [b.com[0] + off[0], b.com[1] + off[1], b.com[2] + off[2]];
                self.sph.vel[idx] = b.vel;
            }
        }
        self.sph.compute_density();
        self.sph.update_stress(dt);
        let accel = self.sph.compute_accels();
        self.sph.integrate_fluid(&accel, dt);
        let pm = self.sph.mass;
        let g = self.sph.gravity;
        for b in &mut self.bodies {
            let mut f = [0.0f64; 3];
            for k in 0..b.offsets.len() {
                let a = accel[b.start + k];
                for ax in 0..3 {
                    f[ax] += pm * a[ax];
                }
            }
            for ax in 0..3 {
                let acc = f[ax] / b.mass + g[ax];
                b.vel[ax] += acc * dt;
                b.com[ax] += b.vel[ax] * dt;
            }
        }
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_peaks_at_origin_and_vanishes_past_support() {
        let h = 0.05;
        assert!(kernel_w(0.0, h) > 0.0);
        assert!(kernel_w(0.0, h) > kernel_w(h, h));
        assert!(kernel_w(h, h) > kernel_w(1.9 * h, h));
        assert_eq!(kernel_w(2.0 * h, h), 0.0);
        assert_eq!(kernel_w(3.0 * h, h), 0.0);
    }

    #[test]
    fn kernel_gradient_points_inward_and_is_zero_at_origin() {
        let h = 0.05;
        assert_eq!(kernel_grad_w([0.0; 3], 0.0, h), [0.0; 3]);
        // For rvec = x_i - x_j along +x, W decreases with r so the
        // gradient points toward j (negative x).
        let g = kernel_grad_w([h, 0.0, 0.0], h, h);
        assert!(g[0] < 0.0);
    }

    #[test]
    fn tait_pressure_is_zero_at_rest_and_rises_under_compression() {
        let p0 = tait_pressure(1000.0, 1000.0, 5.0e4, 7.0);
        let p_hi = tait_pressure(1050.0, 1000.0, 5.0e4, 7.0);
        assert!((p0).abs() < 1e-6);
        assert!(p_hi > 0.0);
        // Free-surface deficit clamps to zero, never negative.
        assert_eq!(tait_pressure(900.0, 1000.0, 5.0e4, 7.0), 0.0);
    }

    #[test]
    fn packed_lattice_density_is_near_rest_density() {
        // A cubic lattice at spacing h with mass = rho0 * h^3 should
        // sample close to the rest density (kernel-consistency check).
        let h = 0.05;
        let rho0 = 1000.0;
        let mut pos = Vec::new();
        for ix in 0..8 {
            for iy in 0..8 {
                for iz in 0..8 {
                    pos.push([ix as f64 * h, iy as f64 * h, iz as f64 * h]);
                }
            }
        }
        let n = pos.len();
        let mut oracle = SphOracle {
            pos,
            vel: vec![[0.0; 3]; n],
            n_fluid: n,
            density: vec![],
            pressure: vec![],
            stress: vec![],
            h,
            mass: rho0 * h * h * h,
            rest_density: rho0,
            stiffness_b: 5.0e4,
            gamma: 7.0,
            viscosity: 0.25,
            gravity: [0.0, -9.81, 0.0],
            domain_min: [-1.0; 3],
            domain_max: [2.0; 3],
            wall_restitution: 0.0,
            friction: 0.0,
        };
        oracle.compute_density();
        // Interior particle (well away from the lattice boundary).
        let interior = 4 * 64 + 4 * 8 + 4;
        let rho = oracle.density[interior];
        assert!(
            (rho - rho0).abs() / rho0 < 0.15,
            "interior density {rho} not near rest {rho0}"
        );
    }

    #[test]
    fn step_keeps_particles_finite_and_in_domain() {
        let h = 0.05;
        let rho0 = 1000.0;
        let mut pos = Vec::new();
        for ix in 0..5 {
            for iy in 0..5 {
                for iz in 0..5 {
                    pos.push([ix as f64 * h, 0.2 + iy as f64 * h, iz as f64 * h]);
                }
            }
        }
        let n = pos.len();
        let mut oracle = SphOracle {
            pos,
            vel: vec![[0.0; 3]; n],
            n_fluid: n,
            density: vec![],
            pressure: vec![],
            stress: vec![],
            h,
            mass: rho0 * h * h * h,
            rest_density: rho0,
            stiffness_b: 5.0e4,
            gamma: 7.0,
            viscosity: 0.3,
            gravity: [0.0, -9.81, 0.0],
            domain_min: [-1.0, 0.0, -1.0],
            domain_max: [1.0, 1.0, 1.0],
            wall_restitution: 0.0,
            friction: 0.0,
        };
        for _ in 0..200 {
            oracle.step(1.0e-4);
        }
        for p in &oracle.pos {
            for (k, &c) in p.iter().enumerate() {
                assert!(c.is_finite(), "non-finite position");
                assert!(
                    c >= oracle.domain_min[k] - 1e-9 && c <= oracle.domain_max[k] + 1e-9,
                    "particle left the domain on axis {k}: {c}"
                );
            }
        }
    }

    #[test]
    fn fluid_rests_on_a_boundary_floor() {
        // A fluid block dropped onto a slab of static boundary
        // particles must come to rest on it, not sink through. The
        // domain clamp is parked far away so the boundary particles —
        // not the clamp — are what hold the fluid up.
        let h = 0.05;
        let rho0 = 1000.0;
        let mut fluid = Vec::new();
        for ix in 0..4 {
            for iy in 0..4 {
                for iz in 0..4 {
                    fluid.push([ix as f64 * h, 0.22 + iy as f64 * h, iz as f64 * h]);
                }
            }
        }
        let n_fluid = fluid.len();
        // Two boundary layers (y = 0.05, 0.10) spanning wider than the
        // fluid footprint.
        let mut pos = fluid;
        for ix in -2..6 {
            for iy in 0..2 {
                for iz in -2..6 {
                    pos.push([ix as f64 * h, 0.05 + iy as f64 * h, iz as f64 * h]);
                }
            }
        }
        let n = pos.len();
        let mut oracle = SphOracle {
            pos,
            vel: vec![[0.0; 3]; n],
            n_fluid,
            density: vec![],
            pressure: vec![],
            stress: vec![],
            h,
            mass: rho0 * h * h * h,
            rest_density: rho0,
            stiffness_b: 5.0e4,
            gamma: 7.0,
            viscosity: 0.3,
            gravity: [0.0, -9.81, 0.0],
            domain_min: [-100.0; 3],
            domain_max: [100.0; 3],
            wall_restitution: 0.0,
            friction: 0.0,
        };
        for _ in 0..400 {
            oracle.step(5.0e-4);
        }
        for p in &oracle.pos[..n_fluid] {
            assert!(p[1].is_finite(), "non-finite fluid particle");
            assert!(
                p[1] > 0.07,
                "fluid sank through the boundary floor: y={}",
                p[1]
            );
        }
        // Boundary particles never moved (started at y ≤ 0.10).
        for (i, p) in oracle.pos[n_fluid..].iter().enumerate() {
            assert!(
                p[1] <= 0.10 + 1e-9,
                "boundary particle {i} moved: y={}",
                p[1]
            );
        }
    }

    #[test]
    fn submerged_body_feels_upward_fluid_force() {
        // A body submerged in a fluid pool must feel a net upward fluid
        // load — it falls markedly slower than free-fall (F3 FSI
        // coupling, CPU oracle). The GPU FSI tests cover a body that
        // fully rises; the small wall-less CPU scene only resolves the
        // direction of the coupling force, not its full magnitude.
        let h = 0.05;
        let rho0 = 1000.0;
        let box_com = [0.0, 0.15, 0.0];
        let mut offsets = Vec::new();
        for ix in -1..=1 {
            for iy in -1..=1 {
                for iz in -1..=1 {
                    offsets.push([ix as f64 * h, iy as f64 * h, iz as f64 * h]);
                }
            }
        }
        // A deep fluid pool resting on the domain floor, confined to
        // its footprint and surrounding the box (box volume excluded).
        let mut fluid = Vec::new();
        for ix in -2..=2 {
            for iy in 0..=8 {
                for iz in -2..=2 {
                    let p = [ix as f64 * h, iy as f64 * h, iz as f64 * h];
                    let in_box = (p[0] - box_com[0]).abs() < 1.5 * h
                        && (p[1] - box_com[1]).abs() < 1.5 * h
                        && (p[2] - box_com[2]).abs() < 1.5 * h;
                    if !in_box {
                        fluid.push(p);
                    }
                }
            }
        }
        let n_fluid = fluid.len();
        let mut pos = fluid;
        for off in &offsets {
            pos.push([
                box_com[0] + off[0],
                box_com[1] + off[1],
                box_com[2] + off[2],
            ]);
        }
        let n = pos.len();
        let sph = SphOracle {
            pos,
            vel: vec![[0.0; 3]; n],
            n_fluid,
            density: vec![],
            pressure: vec![],
            stress: vec![],
            h,
            mass: rho0 * h * h * h,
            rest_density: rho0,
            stiffness_b: 5.0e4,
            gamma: 7.0,
            viscosity: 0.3,
            gravity: [0.0, -9.81, 0.0],
            // Tight domain — the clamp confines the pool to its
            // footprint so it stays a deep pool that submerges the box.
            domain_min: [-0.12, 0.0, -0.12],
            domain_max: [0.12, 0.6, 0.12],
            wall_restitution: 0.0,
            friction: 0.0,
        };
        // A very light body (~60 kg/m³, styrofoam-like). The simple
        // sample-based coupling under-estimates buoyancy magnitude on
        // a small wall-less CPU scene, so a low body density is used
        // here for a clear rise; the GPU FSI tests cover a denser box.
        let box_mass = 0.06;
        let mut fsi = SphFsiOracle {
            sph,
            bodies: vec![SphOracleBody {
                mass: box_mass,
                com: box_com,
                vel: [0.0; 3],
                offsets,
                start: n_fluid,
            }],
        };
        let y0 = fsi.bodies[0].com[1];
        for _ in 0..450 {
            fsi.step_fsi(2.0e-4);
        }
        let y1 = fsi.bodies[0].com[1];
        assert!(y1.is_finite(), "box com non-finite");
        // Free-fall reference (no fluid force) over the same schedule.
        let mut ff_y = y0;
        let mut ff_v = 0.0;
        for _ in 0..450 {
            ff_v += -9.81 * 2.0e-4;
            ff_y += ff_v * 2.0e-4;
        }
        assert!(
            y1 > ff_y + 0.015,
            "submerged body felt no upward fluid force: y1={y1} free_fall={ff_y}"
        );
    }

    #[test]
    fn granular_stress_obeys_drucker_prager_yield() {
        // F4 — verify the regularised perfectly-plastic Drucker-Prager
        // constitutive law: a compressed granular block under simple
        // shear develops a deviatoric stress that (a) is bounded by the
        // yield τ_y = friction·pressure, (b) is aligned with the strain
        // rate so it dissipates rather than pumps energy, and (c) grows
        // monotonically toward the yield surface as the shear rate
        // rises. A frictionless run develops no shear stress at all.
        let h = 0.05;
        let rho0 = 1000.0;
        // Packed tighter than h so the density exceeds rho0 and there
        // is a real confining pressure.
        let sp = 0.95 * h;
        let mut pos = Vec::new();
        for ix in 0..5 {
            for iy in 0..5 {
                for iz in 0..5 {
                    pos.push([ix as f64 * sp, iy as f64 * sp, iz as f64 * sp]);
                }
            }
        }
        let n = pos.len();
        let make = |friction: f64| SphOracle {
            pos: pos.clone(),
            vel: vec![[0.0; 3]; n],
            n_fluid: n,
            density: vec![],
            pressure: vec![],
            stress: vec![],
            h,
            mass: rho0 * h * h * h,
            rest_density: rho0,
            stiffness_b: 5.0e4,
            gamma: 7.0,
            viscosity: 0.25,
            gravity: [0.0, 0.0, 0.0],
            domain_min: [-10.0; 3],
            domain_max: [10.0; 3],
            wall_restitution: 0.0,
            friction,
        };
        let interior = 2 * 25 + 2 * 5 + 2;
        let snorm = |s: [f64; 6]| {
            (s[0] * s[0]
                + s[1] * s[1]
                + s[2] * s[2]
                + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
                .sqrt()
        };
        // ‖S‖ for a granular block under a simple shear vx = shear·y.
        let sheared_stress = |friction: f64, shear: f64| -> ([f64; 6], f64) {
            let mut sand = make(friction);
            sand.compute_density();
            for i in 0..n {
                sand.vel[i] = [shear * sand.pos[i][1], 0.0, 0.0];
            }
            sand.update_stress(2.0e-4);
            (sand.stress[interior], sand.pressure[interior])
        };

        let (s_lo, p) = sheared_stress(0.6, 50.0);
        let (s_hi, _) = sheared_stress(0.6, 500.0);
        assert!(
            p > 0.0,
            "compressed block carries no confining pressure: p={p}"
        );
        let tau_y = 0.6 * p;
        let lo = snorm(s_lo);
        let hi = snorm(s_hi);
        // (a) Bounded by the Drucker-Prager yield at both rates.
        assert!(
            lo <= tau_y * 1.001 && hi <= tau_y * 1.001,
            "deviatoric stress exceeds the yield τ_y={tau_y}: lo={lo} hi={hi}"
        );
        // (b) Dissipative — Sxy shares the sign of the imposed shear.
        assert!(
            s_lo[3] > 0.0 && s_hi[3] > 0.0,
            "shear stress opposes neither rate (not dissipative): {} {}",
            s_lo[3],
            s_hi[3]
        );
        // (c) Monotone toward the yield surface as the shear rate rises.
        assert!(
            hi > lo && hi > 0.6 * tau_y && lo > 0.0,
            "stress is not monotone toward the yield: lo={lo} hi={hi} τ_y={tau_y}"
        );

        // Frictionless material carries no shear stress regardless.
        let (water_s, _) = sheared_stress(0.0, 500.0);
        assert_eq!(
            water_s, [0.0; 6],
            "frictionless material should carry no granular shear stress"
        );
    }
}
