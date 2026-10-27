use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DriftBackend {
    Eager,
    Lazy,
    NativeGeometry,
    NativeLazyGeometry,
}

impl DriftBackend {
    pub(super) const fn compiled() -> Self {
        if cfg!(feature = "native-lazy-geometry-drift") {
            Self::NativeLazyGeometry
        } else if cfg!(feature = "native-geometry-drift") {
            Self::NativeGeometry
        } else if cfg!(feature = "lazy-paired-drift") {
            Self::Lazy
        } else {
            Self::Eager
        }
    }

    pub(super) fn select(paired: bool, local: bool, native_sum: bool, coordinates: bool) -> Self {
        if paired && !local && !native_sum && !coordinates {
            Self::compiled()
        } else {
            Self::Eager
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Eager => "eager-anchors",
            Self::Lazy => "lazy-anchors",
            Self::NativeGeometry => "native-geometry-sums",
            Self::NativeLazyGeometry => "native-geometry-lazy-anchors",
        }
    }

    pub(super) const fn build_label(self) -> &'static str {
        match self {
            Self::Eager => "eager-anchors",
            Self::Lazy => "lazy-anchors-unrestricted-storage-only",
            Self::NativeGeometry => "native-geometry-unrestricted-storage-only",
            Self::NativeLazyGeometry => "native-geometry-lazy-anchors-unrestricted-storage-only",
        }
    }

    pub(super) fn required_features(self) -> wgpu::Features {
        if matches!(self, Self::NativeGeometry | Self::NativeLazyGeometry) {
            wgpu::Features::SHADER_F64
        } else {
            wgpu::Features::empty()
        }
    }

    pub(super) fn validate_features(self, features: wgpu::Features) -> Result<()> {
        if features.contains(self.required_features()) {
            Ok(())
        } else {
            Err(Error::Gpu(
                "native geometry drift requires SHADER_F64 on the supplied device".into(),
            ))
        }
    }

    pub(super) fn source(self, source: &str, paired: bool) -> String {
        let source = unrestricted_drift_source(source);
        if !paired {
            return source;
        }
        match self {
            Self::Eager => source,
            Self::Lazy => lazy_anchors::source(&source),
            Self::NativeGeometry => native_geometry_source(&source),
            Self::NativeLazyGeometry => lazy_anchors::source(&native_geometry_source(&source)),
        }
    }
}

fn native_geometry_source(source: &str) -> String {
    assert_eq!(source.matches("fn position_sum(").count(), 1);
    assert_eq!(source.matches("fn coordinate_sum(").count(), 1);
    let source = format!(
        "{}\n{}\n{}",
        source
            .replace("fn position_sum(", "fn storage_position_sum(")
            .replace("fn coordinate_sum(", "fn storage_coordinate_sum("),
        include_str!("../shaders/featherstone_native_sum.wgsl"),
        "fn position_sum(child: u32, a: f32, b: f32) -> vec2<f32> { let high=position_round(child,a+b); return native_two_sum_with_high(a,b,high); }\nfn coordinate_sum(coord: u32, a: f32, b: f32) -> vec2<f32> { let high=coordinate_round(coord,a+b); let result=native_two_sum_with_high(a,b,high); atomicStore(&coordinate_error[coord],bitcast<u32>(result.y)); return result; }"
    );
    let source=source.replace("let high=position_round(child,a+b); return native_two_sum_with_high(a,b,high);",
        "let high=position_round(child,a+b); let result=native_two_sum_with_high(a,b,high); return vec2<f32>(high,position_round(child,result.y));");
    source.replace("fn coordinate_sum(","fn native_coordinate_sum(")
        .replace("fn storage_coordinate_sum(","fn coordinate_sum(")
        .replace("fn position_sum(child: u32, a: f32, b: f32) -> vec2<f32> { let high=",
        "fn position_sum(child: u32, a: f32, b: f32) -> vec2<f32> { if !fk_pair_rounding { return storage_position_sum(child,a,b); } let high=")
}

#[test]
fn native_geometry_lazy_source_contract() {
    let base = rates_tests::shader_source();
    for capacity in [1, 2, 6, 32] {
        let base = base.replace(
            "const MAX_JOINTS_PER_ART: u32 = 32u;",
            &format!("const MAX_JOINTS_PER_ART: u32 = {capacity}u;"),
        );
        let reference = lazy_anchors::source(&DriftBackend::NativeGeometry.source(&base, true));
        let candidate = unrestricted_drift_source(&native_sum_test_source(
            &lazy_anchors::source(&base),
            true,
            FeatherstonePasses::UnrestrictedDriftNativeGeometryLazyAnchors,
        ));
        assert_eq!(reference, candidate);
        assert_eq!(
            DriftBackend::NativeLazyGeometry.source(&base, true),
            candidate
        );
        assert_eq!(
            DriftBackend::NativeLazyGeometry.source(&base, false),
            unrestricted_drift_source(&base)
        );
        assert!(
            candidate.contains("if !fk_pair_rounding { return storage_position_sum(child,a,b); }")
        );
    }
}

#[test]
fn native_geometry_shader_matches_verified_test_composition() {
    let base = rates_tests::shader_source();
    for capacity in [1, 2, 6, 32] {
        let source = base.replace(
            "const MAX_JOINTS_PER_ART: u32 = 32u;",
            &format!("const MAX_JOINTS_PER_ART: u32 = {capacity}u;"),
        );
        let reference = unrestricted_drift_source(&native_sum_test_source(
            &source,
            true,
            FeatherstonePasses::UnrestrictedDriftNativeGeometry,
        ));
        let candidate = DriftBackend::NativeGeometry.source(&source, true);
        assert_eq!(reference, candidate);
        assert_eq!(
            DriftBackend::NativeGeometry.source(&source, false),
            unrestricted_drift_source(&source)
        );
        let module = naga::front::wgsl::parse_str(&candidate).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
fn native_geometry_backend_admission_and_device_requirements() {
    for bits in 0..16u32 {
        let backend =
            DriftBackend::select(bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0);
        assert_eq!(
            backend,
            if bits == 1 {
                DriftBackend::compiled()
            } else {
                DriftBackend::Eager
            }
        );
    }
    for backend in [
        DriftBackend::Eager,
        DriftBackend::Lazy,
        DriftBackend::NativeGeometry,
        DriftBackend::NativeLazyGeometry,
    ] {
        assert!(
            backend
                .validate_features(wgpu::Features::SHADER_F64)
                .is_ok()
        );
        assert_eq!(
            backend.validate_features(wgpu::Features::empty()).is_err(),
            matches!(
                backend,
                DriftBackend::NativeGeometry | DriftBackend::NativeLazyGeometry
            )
        );
    }
}

#[test]
fn native_geometry_acceleration_source_contract() {
    let original = rates_tests::shader_source();
    let original = original.replace(
        "const MAX_JOINTS_PER_ART: u32 = 32u;",
        "const MAX_JOINTS_PER_ART: u32 = 6u;",
    );
    let candidate = native_sum_test_source(
        &original,
        true,
        FeatherstonePasses::PairedAccelerationNativeGeometry,
    );
    assert_eq!(candidate, native_geometry_source(&original));
    let switched = switched_rigorous_acceleration_source(&candidate);
    assert_eq!(
        switched.replace(
            "if uniforms.joint_damping == 0.0 || !fk_pair_rounding",
            "if !fk_pair_rounding"
        ),
        fixed_rigorous_acceleration_source(&candidate).replace("uniforms.joint_damping", "0.0")
    );
    let module = naga::front::wgsl::parse_str(&rates::dispatch_source(&switched, true)).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    for source in [&original, &candidate] {
        let fixed = fixed_rigorous_acceleration_source(source);
        let source = rates::split_acceleration_source(source, true);
        assert_eq!(fixed, source.replace("uniforms.rigorous_d_solve", "1u"));
        assert!(!source.contains("// Reconstruct body poses and world COM velocities."));
        assert!(!source.contains("// ── Integrate q, qd"));
        assert!(source.contains("velocity_delta[dof+d]=q_dd_arr[k][d]"));
        for shader in [&source, &fixed] {
            let module =
                naga::front::wgsl::parse_str(&rates::dispatch_source(shader, true)).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
    }
}
