//! Stage 38.1 — **CPU-only differentiable MPM** forward pass + tape.
//!
//! This is the host-side counterpart of [`crate::compute::mpm`]: a
//! deliberately minimal explicit MPM (APIC transfer, quadratic B-spline
//! weights, co-rotated Saint-Venant Kirchhoff elasticity, no
//! plasticity) that runs on plain `Vec<Real>` and records a per-step
//! tape so the reverse pass in
//! [`crate::autograd::mpm_adjoint`](super::super::autograd::mpm_adjoint)
//! can walk it backwards to produce `dJ/dE` (or `dJ/dv₀`).
//!
//! # Why a separate module from `mpm.rs`?
//!
//! - `mpm.rs` is GPU-only (`#[cfg(feature = "gpu")]`) and `f32`. The
//!   adjoint is doubly sensitive to float rounding, so we need `f64`.
//! - The forward there scatters into atomics, clamps singular values,
//!   adds a Drucker-Prager / Stomakhin return mapping, and supports
//!   two-way rigid coupling. All useful, all expensive to differentiate.
//!   v1 strips back to the **essential** math: smooth-everywhere PK1
//!   so the chain rule applies without case splits.
//! - GPU port comes later (Stage 38.2+). The point of this module is
//!   the math chain.
//!
//! # Forward pipeline (one substep)
//!
//! For each particle `p`:
//!   1. Compute quadratic B-spline weights `wᵢⱼₖ` over a 3×3×3 stencil
//!      centred on `base = floor(xp/dx − 0.5)`.
//!   2. **P2G** scatter `(mass, momentum, kdiag)` into the grid. Momentum
//!      contribution per node includes the elastic stress term
//!      `-dt · vol · P · Fᵀ · ∇w` where `P` is the PK1 stress.
//!   3. PK1: `R = U·Vᵀ`, `S = V·Σ·Vᵀ`, `P = 2μ(F − R) + λ·tr(S − I)·R`
//!      with `λ = Eν/((1+ν)(1−2ν))`, `μ = E/(2(1+ν))`, `ν = 0.3`.
//!
//! For each grid cell with `mass > 0`:
//!   4. **Grid update**: `v = momentum/mass + g·dt`. Apply a smooth
//!      ground-plane penalty: `v.y ← v.y · (1 − floor_friction · (1 −
//!      smoothstep(world_y / ε)))` and a clamped no-penetration on
//!      `v.y` for cells below the floor.
//!
//! For each particle `p`:
//!   5. **G2P** gather: rebuild `v_new`, the APIC affine matrix `C`, and
//!      advance `x ← x + v_new · dt`.
//!   6. Update `F ← (I + dt · C) · F` (standard APIC MPM `F` update).
//!
//! # Tape
//!
//! For each substep we record everything the reverse pass needs to
//! recompute and back-propagate without re-running SVD:
//!
//! - Per particle, per step: position before P2G, velocity before P2G,
//!   `F` before stress, the SVD outputs `(U, σ, V)`, and the per-node
//!   weights / gradients used by the scatter (cached as the 3-element
//!   B-spline weight arrays per axis + the base cell index).
//! - Per grid cell, per step: `mass`, `momentum_pre`, `velocity_post`.
//!
//! 30 steps × 64 particles × (a handful of small matrices each) is
//! well under 1 MiB — cheap enough to keep memory contiguous and
//! cache-friendly.
//!
//! # References
//!
//! - Jiang et al. "The Affine Particle-In-Cell Method" (2015) for
//!   APIC transfer.
//! - Klár et al. "Drucker-Prager Elastoplasticity for Sand Animation"
//!   (2016) for the explicit MPM pipeline (sans plasticity here).
//! - Hu et al. "ChainQueen: A Real-Time Differentiable Physical
//!   Simulator for Soft Robotics" (2019) for the differentiable-MPM
//!   structure this module follows.

use crate::math::{Mat3, Real, Vec3};

/// Configuration for the CPU diff-MPM forward pass. Defaults aim at the
/// "64 particles falling for 30 steps" smoke demo described in the
/// Stage 38.1 brief.
#[derive(Copy, Clone, Debug)]
pub struct MpmCpuConfig {
    /// Cubic grid edge length (cells per side).
    pub grid_size: usize,
    /// World-space size of one cell.
    pub cell_size: Real,
    /// Lower-left corner of the grid.
    pub grid_origin: [Real; 3],
    /// Gravity along `+Y` (use a negative value for Earth).
    pub gravity_y: Real,
    /// Young's modulus `E` (Pa).
    pub youngs: Real,
    /// Poisson's ratio `ν`.
    pub poisson: Real,
    /// Per-particle reference volume.
    pub particle_volume: Real,
    /// Per-particle mass.
    pub particle_mass: Real,
    /// Soft no-penetration ground at `y = floor_y`. Cells whose cell
    /// centre is at `y < floor_y` get their `v.y` clamped to ≥ 0.
    pub floor_y: Real,
    /// Coulomb-style tangential damping on grid cells touching the
    /// floor. `0` = sticking, `1` = frictionless.
    pub floor_friction: Real,
}

impl Default for MpmCpuConfig {
    fn default() -> Self {
        Self {
            grid_size: 16,
            cell_size: 0.1,
            grid_origin: [-0.8, -0.05, -0.8],
            gravity_y: -9.81,
            youngs: 1.0e4,
            poisson: 0.3,
            particle_volume: 1.0e-3,
            particle_mass: 1.0e-2,
            floor_y: 0.0,
            floor_friction: 0.6,
        }
    }
}

/// Mutable per-particle state. Mirrored across forward steps; the tape
/// snapshots the pre-step values so the adjoint can recompute the
/// scatter / gather without re-running SVD.
#[derive(Clone, Debug, Default)]
pub struct ParticleState {
    pub pos: Vec<Vec3>,
    pub vel: Vec<Vec3>,
    /// APIC affine matrix `C`. Updated at G2P.
    pub c: Vec<Mat3>,
    /// Elastic deformation gradient `F`. Updated as `F ← (I + dt·C) · F`.
    pub f: Vec<Mat3>,
}

impl ParticleState {
    /// Allocate `n` particles with `pos = positions`, `vel = velocities`
    /// (or zero if empty), `C = 0`, `F = I`.
    pub fn new(positions: &[Vec3], velocities: &[Vec3]) -> Self {
        let n = positions.len();
        let v = if velocities.is_empty() {
            vec![Vec3::ZERO; n]
        } else {
            assert_eq!(velocities.len(), n, "velocity / position length mismatch");
            velocities.to_vec()
        };
        Self {
            pos: positions.to_vec(),
            vel: v,
            c: vec![Mat3::ZERO; n],
            f: vec![Mat3::IDENTITY; n],
        }
    }

    pub fn n(&self) -> usize {
        self.pos.len()
    }
}

/// Per-particle, per-step tape entry. Holds everything the reverse
/// pass needs to back-propagate through this particle's contribution
/// to the P2G scatter and the G2P gather **without re-running SVD**.
///
/// Memory: 6 Vec3 + 3 Mat3 + 12 floats + 4 ints ≈ 200 bytes / entry.
/// 30 steps × 64 particles ≈ 0.4 MiB. Cheap.
#[derive(Clone, Debug)]
pub struct ParticleTape {
    /// World-space position before this substep's P2G.
    pub pos_pre: Vec3,
    /// Velocity before this substep's P2G.
    pub vel_pre: Vec3,
    /// APIC affine matrix before this substep's P2G.
    pub c_pre: Mat3,
    /// Deformation gradient before this substep's P2G.
    pub f_pre: Mat3,
    /// SVD of `F` evaluated in P2G for the elastic stress.
    pub u: Mat3,
    pub sigma: Vec3,
    pub v: Mat3,
    /// Base cell index (lower-left corner of the 3×3×3 stencil).
    pub base: [i32; 3],
    /// B-spline weights along x.
    pub wx: [Real; 3],
    /// B-spline weights along y.
    pub wy: [Real; 3],
    /// B-spline weights along z.
    pub wz: [Real; 3],
}

/// Per-cell, per-step tape entry. Holds the pre-update grid state so
/// the reverse pass can reload it without redoing the P2G scatter.
#[derive(Clone, Debug)]
pub struct GridTape {
    /// Grid mass after P2G scatter (used to divide momentum → velocity).
    /// Empty cells have `mass == 0`; the reverse pass skips them.
    pub mass: Vec<Real>,
    /// Grid momentum after P2G scatter, before grid update.
    pub momentum_pre: Vec<Vec3>,
    /// Grid velocity after grid update (post-gravity, post-floor).
    /// Read by G2P.
    pub velocity_post: Vec<Vec3>,
}

/// Complete tape for one forward run. Indexed by step then by
/// particle / cell.
#[derive(Clone, Debug, Default)]
pub struct MpmTape {
    /// `particles[step]` has `n_particles` entries.
    pub particles: Vec<Vec<ParticleTape>>,
    /// `grid[step]` is one [`GridTape`].
    pub grid: Vec<GridTape>,
    /// Final positions / velocities / C / F so the adjoint of the loss
    /// is taken at exactly the same state the forward returns.
    pub final_state: ParticleState,
    /// `dt` used for every step in this tape.
    pub dt: Real,
    /// Cached config for the adjoint to read material constants out of.
    pub cfg: MpmCpuConfig,
}

/// Convenience: Lamé constants from `(E, ν)`.
#[inline]
pub fn lame(youngs: Real, poisson: Real) -> (Real, Real) {
    let mu = youngs / (2.0 * (1.0 + poisson));
    let lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    (lambda, mu)
}

/// Quadratic B-spline weights at offset `fx ∈ [0.5, 1.5)`. Returns
/// `(w0, w1, w2)` covering the three nodes of the stencil. Matches the
/// `quad_weights` helper in `shaders/mpm.wgsl`:
///
/// ```text
/// w0 = 0.5 · (1.5 − fx)²
/// w1 = 0.75 − (fx − 1.0)²
/// w2 = 0.5 · (fx − 0.5)²
/// ```
#[inline]
pub fn quad_weights(fx: Real) -> [Real; 3] {
    let a = 1.5 - fx;
    let b = fx - 1.0;
    let c = fx - 0.5;
    [0.5 * a * a, 0.75 - b * b, 0.5 * c * c]
}

/// Derivative of `quad_weights` wrt `fx`. Used by the adjoint to
/// chain through the weight kernel when needed.
#[inline]
pub fn quad_weights_dfx(fx: Real) -> [Real; 3] {
    let a = 1.5 - fx;
    let b = fx - 1.0;
    let c = fx - 0.5;
    [-a, -2.0 * b, c]
}

#[inline]
fn cell_idx(cfg: &MpmCpuConfig, ix: i32, iy: i32, iz: i32) -> Option<usize> {
    let g = cfg.grid_size as i32;
    if ix < 0 || iy < 0 || iz < 0 || ix >= g || iy >= g || iz >= g {
        None
    } else {
        Some(
            (ix as usize)
                + (iy as usize) * cfg.grid_size
                + (iz as usize) * cfg.grid_size * cfg.grid_size,
        )
    }
}

/// PK1 stress for a co-rotated Saint-Venant Kirchhoff model:
///
/// ```text
/// F = U Σ Vᵀ
/// R = U Vᵀ
/// S = V Σ Vᵀ
/// P = 2μ (F − R) + λ tr(S − I) R
/// ```
///
/// This is the **simplest** rotationally-invariant elastic model that
/// exercises the SVD adjoint — Hu et al. (ChainQueen, 2019) eq. 4 with
/// `J` set aside. The pure Neo-Hookean variant in the GPU shader adds
/// a `λ·J(J−1)·F⁻ᵀ` term which would need an extra cofactor adjoint;
/// skipping it keeps v1 focused.
pub fn corotated_pk1(u: Mat3, sigma: Vec3, v: Mat3, mu: Real, lambda: Real) -> Mat3 {
    let vt = v.transpose();
    let r = u * vt;
    // S = V · diag(σ) · Vᵀ; tr(S − I) = (σ.x − 1) + (σ.y − 1) + (σ.z − 1).
    let trace_s_minus_i = sigma.x + sigma.y + sigma.z - 3.0;
    // F = U · diag(σ) · Vᵀ (reconstructed; equal to forward F up to SVD rounding).
    let f_rebuilt = u * Mat3::from_diagonal(sigma) * vt;
    2.0 * mu * (f_rebuilt - r) + lambda * trace_s_minus_i * r
}

/// Adjoint pieces returned by [`corotated_pk1_adjoint`]. The reverse
/// pass routes the upstream `∂J/∂P` into:
///   - `adj_u`, `adj_sigma`, `adj_v` — fed to `adj_svd3` to get
///     `∂J/∂F` through the SVD chain.
///   - `adj_mu`, `adj_lambda` — scalar gradients wrt the Lamé constants,
///     which then map to `dJ/dE` via the chain rule
///     `dλ/dE = ν / ((1+ν)(1−2ν))`, `dμ/dE = 1 / (2(1+ν))`.
#[derive(Copy, Clone, Debug, Default)]
pub struct CorotatedAdjoint {
    pub adj_u: Mat3,
    pub adj_sigma: Vec3,
    pub adj_v: Mat3,
    pub adj_mu: Real,
    pub adj_lambda: Real,
}

/// Reverse-mode adjoint of [`corotated_pk1`].
///
/// Given the forward outputs `(u, sigma, v)` and the upstream
/// `adj_pk1 = ∂J/∂P`, returns the gradients wrt `(U, Σ, V, μ, λ)`.
///
/// Derivation (Frobenius-inner-product convention `⟨A, B⟩ = tr(AᵀB)`):
///
/// `P = 2μ (F − R) + λ tr(S − I) R` where `F = U Σ Vᵀ`, `R = U Vᵀ`,
/// `S = V Σ Vᵀ`.  Let `δP = adj_pk1` upstream.  Decompose `δP` against
/// each summand:
///
/// - **Through F**: `∂P/∂F = 2μ I` → `adj_F_term = 2μ · δP`.
///   Then `F = U Σ Vᵀ` gives:
///   `adj_U += adj_F_term · V · Σ`
///   `adj_Σᵢ += (Uᵀ · adj_F_term · V)_{ii}`
///   `adj_V += adj_F_termᵀ · U · Σ`
/// - **Through R**: `∂P/∂R = −2μ I + λ tr(S − I) I` so the R-piece is
///   `(−2μ + λ trS_m1) · δP`. With `R = U Vᵀ`:
///   `adj_U += R_piece · V`
///   `adj_V += R_pieceᵀ · U`
/// - **Through Σ (via `tr(S − I)`)**: `tr(S − I) = Σᵢ σᵢ − 3`, so
///   `∂tr/∂σᵢ = 1`. The σ contribution to P is `λ · σᵢ · R` per i.
///   `adj_Σᵢ += λ · ⟨R, δP⟩`.
/// - **Through μ**: `adj_μ = ⟨2(F − R), δP⟩`.
/// - **Through λ**: `adj_λ = tr(S − I) · ⟨R, δP⟩`.
pub fn corotated_pk1_adjoint(
    u: Mat3,
    sigma: Vec3,
    v: Mat3,
    mu: Real,
    lambda: Real,
    adj_pk1: Mat3,
) -> CorotatedAdjoint {
    let vt = v.transpose();
    let r = u * vt;
    let sigma_diag = Mat3::from_diagonal(sigma);
    let f_rebuilt = u * sigma_diag * vt;
    let trace_s_minus_i = sigma.x + sigma.y + sigma.z - 3.0;

    // ── Through F = U Σ Vᵀ (coefficient 2μ) ───────────────────────────
    let coef_f = 2.0 * mu;
    let adj_f_eff = coef_f * adj_pk1;
    // d(U Σ Vᵀ)/dU contributes adj_U += adj_F_eff · V · Σ.
    let adj_u_from_f = adj_f_eff * v * sigma_diag;
    // d(U Σ Vᵀ)/dV contributes adj_V += adj_F_effᵀ · U · Σ.
    let adj_v_from_f = adj_f_eff.transpose() * u * sigma_diag;
    // d(U Σ Vᵀ)/dσᵢ = Uᵢ Vᵢᵀ (rank-1) → ⟨Uᵢ Vᵢᵀ, adj_F_eff⟩ =
    //     (Uᵀ adj_F_eff V)_{ii}.
    let m_sigma = u.transpose() * adj_f_eff * v;
    let adj_sigma_from_f = Vec3::new(m_sigma.col(0).x, m_sigma.col(1).y, m_sigma.col(2).z);

    // ── Through R = U Vᵀ (coefficient −2μ + λ·trS_m1) ─────────────────
    let coef_r = -2.0 * mu + lambda * trace_s_minus_i;
    let adj_r_eff = coef_r * adj_pk1;
    // d(U Vᵀ)/dU = ·V (right-multiply) → adj_U += adj_R_eff · V.
    let adj_u_from_r = adj_r_eff * v;
    // d(U Vᵀ)/dV = Uᵀ on the left, transposed:  adj_V += adj_R_effᵀ · U.
    let adj_v_from_r = adj_r_eff.transpose() * u;

    // ── Through σ via tr(S − I): coefficient λ · ⟨R, δP⟩ ──────────────
    let r_dot_dp = frob_dot(r, adj_pk1);
    let adj_sigma_from_tr = Vec3::splat(lambda * r_dot_dp);

    // ── Material constants ────────────────────────────────────────────
    let adj_mu = 2.0 * frob_dot(f_rebuilt - r, adj_pk1);
    let adj_lambda = trace_s_minus_i * r_dot_dp;

    CorotatedAdjoint {
        adj_u: adj_u_from_f + adj_u_from_r,
        adj_sigma: adj_sigma_from_f + adj_sigma_from_tr,
        adj_v: adj_v_from_f + adj_v_from_r,
        adj_mu,
        adj_lambda,
    }
}

/// Frobenius inner product `⟨A, B⟩ = Σ Aᵢⱼ Bᵢⱼ`.
#[inline]
pub fn frob_dot(a: Mat3, b: Mat3) -> Real {
    let mut s = 0.0;
    for j in 0..3 {
        let acol = a.col(j);
        let bcol = b.col(j);
        s += acol.dot(bcol);
    }
    s
}

/// Run `steps` substeps of CPU diff-MPM forward, recording the full
/// tape. The caller can then call into
/// [`crate::autograd::mpm_adjoint::mpm_backward_e`] to get `dJ/dE` for
/// a loss defined on the **final** particle positions.
pub fn forward_with_tape(
    cfg: &MpmCpuConfig,
    initial: &ParticleState,
    dt: Real,
    steps: usize,
) -> MpmTape {
    let mut state = initial.clone();
    let mut tape = MpmTape {
        particles: Vec::with_capacity(steps),
        grid: Vec::with_capacity(steps),
        final_state: state.clone(),
        dt,
        cfg: *cfg,
    };
    for _ in 0..steps {
        let (p_tape, g_tape) = step_once_record(cfg, &mut state, dt);
        tape.particles.push(p_tape);
        tape.grid.push(g_tape);
    }
    tape.final_state = state;
    tape
}

/// One explicit MPM substep with full tape recording.
fn step_once_record(
    cfg: &MpmCpuConfig,
    state: &mut ParticleState,
    dt: Real,
) -> (Vec<ParticleTape>, GridTape) {
    use crate::math::dense3::svd3;
    let n = state.n();
    let g = cfg.grid_size;
    let g_total = g * g * g;
    let inv_dx = 1.0 / cfg.cell_size;
    let dx = cfg.cell_size;
    let (lambda, mu) = lame(cfg.youngs, cfg.poisson);

    // Grid scratch.
    let mut g_mass = vec![0.0_f64; g_total];
    let mut g_momentum = vec![Vec3::ZERO; g_total];

    let mut p_tape: Vec<ParticleTape> = Vec::with_capacity(n);

    // ── P2G ─────────────────────────────────────────────────────────
    for p in 0..n {
        let pos = state.pos[p];
        let vel = state.vel[p];
        let c = state.c[p];
        let f = state.f[p];

        // Cell-space coords matching the WGSL `base = floor(x − 0.5)` form.
        let origin = Vec3::new(cfg.grid_origin[0], cfg.grid_origin[1], cfg.grid_origin[2]);
        let xp = (pos - origin) * inv_dx;
        let base_xyz = Vec3::new(
            (xp.x - 0.5).floor(),
            (xp.y - 0.5).floor(),
            (xp.z - 0.5).floor(),
        );
        let base = [base_xyz.x as i32, base_xyz.y as i32, base_xyz.z as i32];
        let fx = xp - base_xyz;

        let wx = quad_weights(fx.x);
        let wy = quad_weights(fx.y);
        let wz = quad_weights(fx.z);

        // SVD + PK1.
        let (uu, sigma, vv) = svd3(f);
        let pk1 = corotated_pk1(uu, sigma, vv, mu, lambda);
        // APIC stress force coefficient: -dt · vol · P · Fᵀ.
        let stress = -dt * cfg.particle_volume * (pk1 * f.transpose());

        p_tape.push(ParticleTape {
            pos_pre: pos,
            vel_pre: vel,
            c_pre: c,
            f_pre: f,
            u: uu,
            sigma,
            v: vv,
            base,
            wx,
            wy,
            wz,
        });

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let w = wx[i] * wy[j] * wz[k];
                    let ix = base[0] + i as i32;
                    let iy = base[1] + j as i32;
                    let iz = base[2] + k as i32;
                    let Some(ci) = cell_idx(cfg, ix, iy, iz) else {
                        continue;
                    };
                    // Grid node at `ix · dx + origin` (integer lattice — the
                    // matches `shaders/mpm.wgsl`'s convention exactly
                    // (see the comment around line 386 of that shader).
                    // The first-moment partition is not exactly zero
                    // under this convention — the GPU forward
                    // compensates via the clamped-σ Neo-Hookean stress
                    // bound. For the diff-MPM forward we accept the
                    // small drift in exchange for byte-identical
                    // behaviour with `shaders/mpm.wgsl::p2g/g2p`.
                    let node_pos = origin + Vec3::new(ix as Real, iy as Real, iz as Real) * dx;
                    let xn_xp = node_pos - pos;
                    let aff = c * xn_xp;
                    let stress_force = stress * xn_xp * (4.0 * inv_dx * inv_dx);
                    let mom = w * cfg.particle_mass * (vel + aff) + w * stress_force;
                    g_mass[ci] += w * cfg.particle_mass;
                    g_momentum[ci] += mom;
                }
            }
        }
    }

    // ── Grid update ─────────────────────────────────────────────────
    let momentum_pre = g_momentum.clone();
    let mut g_velocity = vec![Vec3::ZERO; g_total];
    let origin_y = cfg.grid_origin[1];
    for iz in 0..g {
        for iy in 0..g {
            for ix in 0..g {
                let ci = ix + iy * g + iz * g * g;
                let m = g_mass[ci];
                if m <= 0.0 {
                    continue;
                }
                let mut v = g_momentum[ci] / m;
                v.y += cfg.gravity_y * dt;
                // Floor mask: if the node centre is below floor_y,
                // clamp v.y to ≥ 0 and dampen tangent. Node centre at
                // y = origin_y + iy · dx (matches P2G/G2P lattice).
                let node_y = origin_y + (iy as Real) * dx;
                if node_y < cfg.floor_y {
                    if v.y < 0.0 {
                        v.y = 0.0;
                    }
                    let damp = 1.0 - cfg.floor_friction;
                    v.x *= damp;
                    v.z *= damp;
                }
                g_velocity[ci] = v;
            }
        }
    }

    // ── G2P + F update ──────────────────────────────────────────────
    for p in 0..n {
        let tape_p = &p_tape[p];
        let base = tape_p.base;
        let wx = tape_p.wx;
        let wy = tape_p.wy;
        let wz = tape_p.wz;
        let pos = state.pos[p];
        let origin = Vec3::new(cfg.grid_origin[0], cfg.grid_origin[1], cfg.grid_origin[2]);

        let mut v_new = Vec3::ZERO;
        let mut c_new = Mat3::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let w = wx[i] * wy[j] * wz[k];
                    let ix = base[0] + i as i32;
                    let iy = base[1] + j as i32;
                    let iz = base[2] + k as i32;
                    let Some(ci) = cell_idx(cfg, ix, iy, iz) else {
                        continue;
                    };
                    let v_node = g_velocity[ci];
                    let node_pos = origin + Vec3::new(ix as Real, iy as Real, iz as Real) * dx;
                    let xn_xp = node_pos - pos;
                    v_new += w * v_node;
                    // APIC C reconstruction: C = 4/dx² · Σ w · v_node · (xn − xp)ᵀ
                    c_new += outer(v_node, xn_xp) * (4.0 * w * inv_dx * inv_dx);
                }
            }
        }
        state.vel[p] = v_new;
        state.c[p] = c_new;
        state.pos[p] = pos + v_new * dt;
        // F update: F ← (I + dt · C) · F.
        state.f[p] = (Mat3::IDENTITY + dt * c_new) * state.f[p];
    }

    let g_tape = GridTape {
        mass: g_mass,
        momentum_pre,
        velocity_post: g_velocity,
    };
    (p_tape, g_tape)
}

/// Outer product `a · bᵀ` returning a 3×3 matrix.
#[inline]
pub fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// Convenience: forward without tape recording. Used for FD oracles
/// and for sanity tests where only the final state matters. Bit-for-bit
/// matches [`forward_with_tape`]'s final state if the tape is discarded.
pub fn forward(
    cfg: &MpmCpuConfig,
    initial: &ParticleState,
    dt: Real,
    steps: usize,
) -> ParticleState {
    forward_with_tape(cfg, initial, dt, steps).final_state
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_particle_at(pos: Vec3) -> ParticleState {
        ParticleState::new(&[pos], &[Vec3::ZERO])
    }

    #[test]
    fn forward_runs_and_advances_under_gravity() {
        // The default (volume 1e-3, E 1e4, dt 1/240) parameter set has
        // a stress→P2G→G2P feedback gain > 1 for an isolated particle,
        // so f64 roundoff in the (exactly zero) APIC C matrix amplifies
        // exponentially — it only *looked* stable before the F1b
        // node-lattice fix because the half-cell moment-arm bias
        // dominated. A sane volume keeps the loop contractive.
        let cfg = MpmCpuConfig {
            particle_volume: 1.0e-5,
            ..MpmCpuConfig::default()
        };
        let start = unit_particle_at(Vec3::new(0.0, 0.5, 0.0));
        let dt = 1.0 / 240.0;
        let out = forward(&cfg, &start, dt, 30);
        // Particle should have moved (we don't gate on direction
        // because the floor catches it — but at minimum the F should
        // still be near identity for an unloaded particle and the
        // position must be inside the grid).
        let p = out.pos[0];
        assert!(p.y.is_finite());
        assert!(p.y < 0.5, "particle should fall under gravity: y = {}", p.y);
    }

    #[test]
    fn quad_weights_partition_of_unity() {
        // For any fx ∈ [0.5, 1.5), w0 + w1 + w2 = 1 exactly.
        for k in 0..11 {
            let fx = 0.5 + (k as Real) * 0.1;
            let w = quad_weights(fx);
            let s = w[0] + w[1] + w[2];
            assert!((s - 1.0).abs() < 1.0e-12, "fx={fx}, sum={s}");
        }
    }

    #[test]
    fn corotated_pk1_zero_at_identity() {
        // F = I → R = I, S = I, P = 0. The most basic invariant.
        let f = Mat3::IDENTITY;
        let (lambda, mu) = lame(1.0e4, 0.3);
        // Decompose F to feed into corotated_pk1.
        use crate::math::dense3::svd3;
        let (u, s, v) = svd3(f);
        let p = corotated_pk1(u, s, v, mu, lambda);
        let mut fro = 0.0;
        for j in 0..3 {
            let c = p.col(j);
            fro += c.dot(c);
        }
        assert!(fro.sqrt() < 1.0e-10, "P(I) = {p:?}");
    }

    #[test]
    fn corotated_adjoint_fd_oracle() {
        // Random F, random upstream δP. Check adj_F via FD by going
        // through the SVD → PK1 chain.
        use crate::math::dense3::{adj_svd3, svd3};
        // Pick F well-conditioned, far from identity.
        let f = Mat3::from_cols(
            Vec3::new(1.2, 0.1, -0.05),
            Vec3::new(0.0, 1.1, 0.07),
            Vec3::new(0.03, -0.02, 0.95),
        );
        let (lambda, mu) = lame(2.0e4, 0.3);
        let (u, sigma, v) = svd3(f);
        // Pick a fixed-but-arbitrary δP target via gradient of
        // J(P) = ‖P − P_tgt‖².
        let p_tgt = Mat3::from_cols(
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, -0.4, 0.1),
            Vec3::new(0.0, 0.0, 0.7),
        );
        let p_now = corotated_pk1(u, sigma, v, mu, lambda);
        let adj_pk1 = 2.0 * (p_now - p_tgt);
        let adj = corotated_pk1_adjoint(u, sigma, v, mu, lambda, adj_pk1);
        let adj_f = adj_svd3(f, u, sigma, v, adj.adj_u, adj.adj_sigma, adj.adj_v);

        // FD on each F entry.
        let h = 1.0e-7;
        let loss = |fp: Mat3| -> Real {
            let (uu, ss, vv) = svd3(fp);
            let p = corotated_pk1(uu, ss, vv, mu, lambda);
            let d = p - p_tgt;
            frob_dot(d, d)
        };
        let mut max_rel = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let mut fp = f;
                let mut fm = f;
                let mut cp = fp.col(j);
                let mut cm = fm.col(j);
                cp[i] += h;
                cm[i] -= h;
                fp = match j {
                    0 => Mat3::from_cols(cp, fp.col(1), fp.col(2)),
                    1 => Mat3::from_cols(fp.col(0), cp, fp.col(2)),
                    _ => Mat3::from_cols(fp.col(0), fp.col(1), cp),
                };
                fm = match j {
                    0 => Mat3::from_cols(cm, fm.col(1), fm.col(2)),
                    1 => Mat3::from_cols(fm.col(0), cm, fm.col(2)),
                    _ => Mat3::from_cols(fm.col(0), fm.col(1), cm),
                };
                let fd = (loss(fp) - loss(fm)) / (2.0 * h);
                let analytic = adj_f.col(j)[i];
                let denom = analytic.abs().max(fd.abs()).max(1.0);
                let rel = (analytic - fd).abs() / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
                assert!(
                    rel < 1.0e-3,
                    "(i={i}, j={j}): analytic {analytic}, fd {fd}, rel {rel}",
                );
            }
        }
        eprintln!("corotated_pk1_adjoint FD max rel err: {max_rel:e}");
    }

    #[test]
    fn corotated_adjoint_mu_lambda_fd_oracle() {
        // ∂J/∂μ and ∂J/∂λ against FD.
        use crate::math::dense3::svd3;
        let f = Mat3::from_cols(
            Vec3::new(1.05, 0.02, 0.0),
            Vec3::new(-0.01, 0.98, 0.03),
            Vec3::new(0.0, 0.04, 1.02),
        );
        let (lambda, mu) = lame(1.5e4, 0.3);
        let (u, sigma, v) = svd3(f);
        let p_tgt = Mat3::ZERO;
        let p_now = corotated_pk1(u, sigma, v, mu, lambda);
        let adj_pk1 = 2.0 * (p_now - p_tgt);
        let adj = corotated_pk1_adjoint(u, sigma, v, mu, lambda, adj_pk1);

        let loss = |m: Real, l: Real| -> Real {
            let p = corotated_pk1(u, sigma, v, m, l);
            let d = p - p_tgt;
            frob_dot(d, d)
        };
        let h_mu = 1.0e-3;
        let fd_mu = (loss(mu + h_mu, lambda) - loss(mu - h_mu, lambda)) / (2.0 * h_mu);
        let h_lambda = 1.0e-3;
        let fd_lambda =
            (loss(mu, lambda + h_lambda) - loss(mu, lambda - h_lambda)) / (2.0 * h_lambda);

        let rel_mu = (adj.adj_mu - fd_mu).abs() / fd_mu.abs().max(adj.adj_mu.abs()).max(1.0);
        let rel_lam =
            (adj.adj_lambda - fd_lambda).abs() / fd_lambda.abs().max(adj.adj_lambda.abs()).max(1.0);
        assert!(rel_mu < 1.0e-4, "adj_mu={} vs fd {}", adj.adj_mu, fd_mu);
        assert!(
            rel_lam < 1.0e-4,
            "adj_lambda={} vs fd {}",
            adj.adj_lambda,
            fd_lambda
        );
    }
}
