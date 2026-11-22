use molla_core::{Error, Result};
use molla_math::{Mat3, Vec3};
use molla_sim::ElasticElement;
use wgpu::util::DeviceExt;

const SOURCE: &str = concat!(
    include_str!("../../shaders/constitutive.wgsl"),
    "\n",
    include_str!("../../shaders/strain_limit.wgsl"),
    "\n",
    include_str!("../../shaders/barrier_materials.wgsl"),
);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Record {
    vertices: [u32; 4],
    rest: [[f32; 4]; 3],
    material: [f32; 4],
    config: [u32; 4],
}

pub(super) struct Prepared {
    records: Vec<Record>,
    offsets: Vec<u32>,
    adjacency: Vec<[u32; 2]>,
    strain_neighbors: Vec<[u32; 2]>,
    strain_count: u32,
    springs: Vec<[u32; 4]>,
    tets: Vec<[u32; 8]>,
}

fn invalid(detail: &str) -> Error {
    Error::Solver(format!("barrier material: {detail}"))
}

impl Prepared {
    pub(super) fn new(
        device: &wgpu::Device,
        positions: &[[f32; 3]],
        elements: &[ElasticElement],
        springs: &[super::BarrierGpuSpring],
        tets: &[([u32; 4], f32)],
    ) -> Result<Option<Self>> {
        if elements.is_empty() {
            return Ok(None);
        }
        let limits = device.limits();
        let count = u32::try_from(elements.len()).map_err(|_| invalid("element count overflow"))?;
        if count.div_ceil(64) > limits.max_compute_workgroups_per_dimension {
            return Err(invalid("dispatch capacity exceeded"));
        }
        for bytes in [
            u64::from(count) * 96,
            u64::from(count) * 400,
            u64::from(count) * 48,
            u64::from(count) * 32,
            (positions.len() as u64 + 1) * 4,
        ] {
            if bytes > u64::from(limits.max_storage_buffer_binding_size)
                || bytes > limits.max_buffer_size
            {
                return Err(invalid("storage capacity exceeded"));
            }
        }
        let strain_count = elements.iter().filter(|e| e.strain_limit().is_some()).count();
        let mut strain_neighbors = vec![[0;2];strain_count];
        if strain_count > 0 {
            let stencils: Vec<_> = elements.iter().map(|e| e.vertices().to_vec())
                .chain(springs.iter().map(|s| vec![s.a,s.b]))
                .chain(tets.iter().map(|t| t.0.to_vec())).collect();
            let faces: Vec<_> = elements.iter().filter(|e| e.strain_limit().is_some())
                .map(|e| { let ids = e.vertices(); [ids[0],ids[1],ids[2]] }).collect();
            let neighbors = molla_sim::elastic_stencil_neighborhoods(positions.len(), &stencils, &faces)?;
            for (i,neighbors) in neighbors.into_iter().enumerate() {
                let start = u32::try_from(strain_neighbors.len()).map_err(|_| invalid("strain adjacency overflow"))?;
                let len = u32::try_from(neighbors.len()).map_err(|_| invalid("strain adjacency overflow"))?;
                strain_neighbors[i] = [start,len];
                for j in neighbors {
                    let j = j as usize;
                    strain_neighbors.push(if j < elements.len() { [0,j as u32] }
                        else if j < elements.len()+springs.len() { [1,(j-elements.len()) as u32] }
                        else { [2,(j-elements.len()-springs.len()) as u32] });
                }
            }
        }
        for bytes in [
            (elements.len() as u64+strain_count as u64)*400,
            strain_neighbors.len() as u64*8,
            if strain_count > 0 { springs.len() as u64*16 } else { 0 },
            if strain_count > 0 { tets.len() as u64*32 } else { 0 },
        ] {
            if bytes > limits.max_buffer_size || bytes > limits.max_storage_buffer_binding_size as u64 {
                return Err(invalid("strain storage capacity exceeded"));
            }
        }
        let positions: Vec<_> = positions
            .iter()
            .map(|p| Vec3::from_array(p.map(f64::from)))
            .collect();
        let mut records = Vec::with_capacity(elements.len());
        let mut offsets = vec![0u32; positions.len() + 1];
        let mut strain_index = 0;
        for e in elements {
            e.validate(&positions)?;
            let mut vertices = [0; 4];
            vertices[..e.vertices().len()].copy_from_slice(e.vertices());
            let mut rest = [[0.0; 4]; 3];
            for (c, column) in rest.iter_mut().enumerate() {
                for (r, component) in column[..3].iter_mut().enumerate() {
                    *component = e.inverse_rest().col(c)[r] as f32;
                }
            }
            let material = [
                e.material().mu as f32,
                e.material().lambda as f32,
                e.measure() as f32,
                e.strain_limit()
                    .map(|l| l.maximum_stretch())
                    .transpose()?
                    .unwrap_or(0.0) as f32,
            ];
            if let Some(limit) = e.strain_limit() {
                rest[0][3] = limit.strain as f32;
                rest[1][3] = e.strain_mass() as f32;
                if rest[0][3] <= 0.0 || rest[1][3] <= 0.0 { return Err(invalid("strain parameters underflow f32")); }
            }
            if !rest
                .iter()
                .flatten()
                .chain(material.iter())
                .all(|v| v.is_finite())
                || material[2] <= 0.0
            {
                return Err(invalid("rest coordinates or coefficients exceed f32 range"));
            }
            let inverse = Mat3::from_cols(
                Vec3::new(rest[0][0] as f64, rest[0][1] as f64, rest[0][2] as f64),
                Vec3::new(rest[1][0] as f64, rest[1][1] as f64, rest[1][2] as f64),
                Vec3::new(rest[2][0] as f64, rest[2][1] as f64, rest[2][2] as f64),
            );
            let origin = positions[vertices[0] as usize];
            let c = if e.dimensions() == 3 {
                positions[vertices[3] as usize] - origin
            } else {
                Vec3::ZERO
            };
            let f = Mat3::from_cols(
                positions[vertices[1] as usize] - origin,
                positions[vertices[2] as usize] - origin,
                c,
            ) * inverse;
            e.material().energy(f, e.dimensions())?;
            if let Some(limit) = e.strain_limit() {
                limit.check(f)?;
                let safe_maximum = material[3] * (1.0 - 16.0 * f32::EPSILON);
                if material[3] <= 1.0
                    || molla_math::constitutive::MembraneStrainLimit::largest_stretch(f)?
                        >= f64::from(safe_maximum)
                {
                    return Err(invalid(
                        "initial membrane exceeds f32 principal-stretch cap",
                    ));
                }
            }
            for &id in e.vertices() {
                offsets[id as usize + 1] = offsets[id as usize + 1]
                    .checked_add(1)
                    .ok_or_else(|| invalid("adjacency count overflow"))?;
            }
            records.push(Record {
                vertices,
                rest,
                material,
                config: [e.dimensions() as u32, e.material().model as u32, strain_index, 0],
            });
            if e.strain_limit().is_some() { strain_index += 1; }
        }
        for i in 1..offsets.len() {
            offsets[i] = offsets[i]
                .checked_add(offsets[i - 1])
                .ok_or_else(|| invalid("adjacency offset overflow"))?;
        }
        let mut cursor = offsets.clone();
        let mut adjacency = vec![[0; 2]; *offsets.last().unwrap() as usize];
        for (i, e) in elements.iter().enumerate() {
            for (local, &id) in e.vertices().iter().enumerate() {
                adjacency[cursor[id as usize] as usize] = [i as u32, local as u32];
                cursor[id as usize] += 1;
            }
        }
        Ok(Some(Self {
            records,
            offsets,
            adjacency,
            strain_neighbors,
            strain_count: strain_count as u32,
            springs: if strain_count == 0 { vec![] } else {
                springs.iter().map(|s| [s.a,s.b,s.rest.to_bits(),s.k.to_bits()]).collect()
            },
            tets: if strain_count == 0 { vec![] } else {
                tets.iter().map(|(ids,k)| [ids[0],ids[1],ids[2],ids[3],k.to_bits(),0,0,0]).collect()
            },
        }))
    }
}

pub(super) struct Materials {
    pipelines: std::collections::HashMap<&'static str, wgpu::ComputePipeline>,
    groups: [wgpu::BindGroup; 4],
    elements: u32,
    vertices: u32,
    strain_count: u32,
}

impl Materials {
    pub(super) fn new(
        device: &wgpu::Device,
        prepared: Prepared,
        uniforms: &wgpu::Buffer,
        pinned: &wgpu::Buffer,
        state: [&wgpu::Buffer; 5],
        outputs: [&wgpu::Buffer; 4],
    ) -> Self {
        let layout = |count: u32, uniform: bool, read_only: bool| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bn.material.layout"),
                entries: &(0..count)
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if uniform {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage { read_only }
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            })
        };
        let layouts = [
            layout(1, true, true),
            layout(7, false, true),
            layout(5, false, true),
            layout(6, false, false),
        ];
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bn.material.pipeline_layout"),
            bind_group_layouts: &layouts.iter().map(Some).collect::<Vec<_>>(),
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bn.material"),
            source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
        });
        let mut pipelines = std::collections::HashMap::new();
        for name in [
            "element_geometry",
            "element_prepare",
            "element_strain",
            "element_gradient",
            "element_product",
            "element_hessian",
            "element_energy",
            "element_sum_energy",
            "element_bound",
            "element_reduce_bound",
        ] {
            pipelines.insert(
                name,
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                }),
            );
        }
        let upload = |label, contents: &[u8]| {
            molla_core::gpu_buffer_init!(
                device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: wgpu::BufferUsages::STORAGE,
                }
            )
        };
        let records = upload(
            "bn.material.elements",
            bytemuck::cast_slice(&prepared.records),
        );
        let offsets = upload(
            "bn.material.offsets",
            bytemuck::cast_slice(&prepared.offsets),
        );
        let adjacency = upload(
            "bn.material.adjacency",
            bytemuck::cast_slice(&prepared.adjacency),
        );
        let elements = prepared.records.len() as u32;
        let strain_count = prepared.strain_count;
        let neighbors = (!prepared.strain_neighbors.is_empty()).then(|| upload("bn.strain.neighbors",bytemuck::cast_slice(&prepared.strain_neighbors)));
        let springs = (!prepared.springs.is_empty()).then(|| upload("bn.strain.springs",bytemuck::cast_slice(&prepared.springs)));
        let tets = (!prepared.tets.is_empty()).then(|| upload("bn.strain.tets",bytemuck::cast_slice(&prepared.tets)));
        let vertices = prepared.offsets.len() as u32 - 1;
        let scratch = |label, count, stride| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size: u64::from(count) * stride,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }
            )
        };
        let cache = scratch("bn.material.tangents", elements+strain_count, 400);
        let work = scratch("bn.material.scratch", elements, 48);
        let group = |index: usize, buffers: &[&wgpu::Buffer]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bn.material.bindings"),
                layout: &layouts[index],
                entries: &buffers
                    .iter()
                    .enumerate()
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        };
        let groups = [
            group(0, &[uniforms]),
            group(1, &[&records, &offsets, &adjacency, pinned,
                neighbors.as_ref().unwrap_or(&records), springs.as_ref().unwrap_or(&records), tets.as_ref().unwrap_or(&records)]),
            group(2, &state),
            group(
                3,
                &[
                    outputs[0], outputs[1], outputs[2], outputs[3], &cache, &work,
                ],
            ),
        ];
        Self {
            pipelines,
            groups,
            elements,
            vertices,
            strain_count,
        }
    }

    fn bind(&self, pass: &mut wgpu::ComputePass<'_>) {
        for (i, group) in self.groups.iter().enumerate() {
            pass.set_bind_group(i as u32, group, &[]);
        }
    }

    fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>, name: &str, groups: u32) {
        pass.set_pipeline(&self.pipelines[name]);
        pass.dispatch_workgroups(groups, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }

    pub(super) fn encode_after(&self, pass: &mut wgpu::ComputePass<'_>, name: &str) -> bool {
        let (first, second, vertex_output) = match name {
            "gradient" => ("element_prepare", "element_gradient", true),
            "hessian_mul" => ("element_product", "element_hessian", true),
            "energy_vertex" => ("element_energy", "element_sum_energy", true),
            "ccd_alpha" => ("element_bound", "element_reduce_bound", false),
            _ => return false,
        };
        self.bind(pass);
        self.dispatch(pass, first, self.elements.div_ceil(64));
        if name == "gradient" && self.strain_count > 0 {
            self.dispatch(pass,"element_strain",self.elements.div_ceil(64));
        }
        self.dispatch(
            pass,
            second,
            if vertex_output {
                self.vertices.div_ceil(64)
            } else {
                1
            },
        );
        true
    }

    pub(super) fn validate(&self, encoder: &mut wgpu::CommandEncoder, scalars: &wgpu::Buffer) {
        encoder.clear_buffer(scalars, 52, Some(4));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bn.material.geometry"),
            timestamp_writes: None,
        });
        self.bind(&mut pass);
        self.dispatch(&mut pass, "element_geometry", self.elements.div_ceil(64));
    }

    pub(super) fn check(status: f32) -> Result<()> {
        let flags = status.to_bits();
        if flags == 0 {
            Ok(())
        } else {
            Err(invalid(&format!(
                "invalid geometry or arithmetic ({flags:#x})"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn shader_parses_and_validates() {
        let module = naga::front::wgsl::parse_str(super::SOURCE)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(super::SOURCE)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[cfg(all(test, feature = "resource-counters"))]
#[path = "materials_tests.rs"]
mod gpu_tests;
