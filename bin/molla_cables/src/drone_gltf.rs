use molla_math::{Mat3, Quat, Vec3};
use serde_json::Value;
use std::sync::OnceLock;

const FILE: &[u8] = include_bytes!("../data/drone/mavic-3.glb");

pub struct Part {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uv: Vec<[f32; 2]>,
    pub indices: Vec<usize>,
    pub texture: usize,
    pub rotor: Option<usize>,
}

pub struct Model {
    pub parts: Vec<Part>,
    pub pivots: [Vec3; 4],
}

fn word(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn source() -> (&'static Value, &'static [u8]) {
    static JSON: OnceLock<Value> = OnceLock::new();
    assert_eq!(&FILE[..4], b"glTF");
    assert_eq!(word(FILE, 4), 2);
    assert_eq!(word(FILE, 8) as usize, FILE.len());
    let end = 20 + word(FILE, 12) as usize;
    assert_eq!(&FILE[end + 4..end + 8], b"BIN\0");
    let json = JSON.get_or_init(|| serde_json::from_slice(&FILE[20..end]).unwrap());
    (json, &FILE[end + 8..])
}

fn view(id: usize) -> &'static [u8] {
    let (json, bytes) = source();
    let v = &json["bufferViews"][id];
    assert_eq!(v["buffer"].as_u64().unwrap_or(0), 0);
    let offset = v["byteOffset"].as_u64().unwrap_or(0) as usize;
    let count = v["byteLength"].as_u64().unwrap() as usize;
    &bytes[offset..offset + count]
}

pub fn textures() -> Vec<image::RgbaImage> {
    let (json, _) = source();
    json["textures"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            let image = &json["images"][t["source"].as_u64().unwrap() as usize];
            assert_eq!(image["mimeType"], "image/png");
            image::load_from_memory(view(image["bufferView"].as_u64().unwrap() as usize))
                .unwrap()
                .to_rgba8()
        })
        .collect()
}

fn accessor(id: usize) -> Vec<Vec<f64>> {
    let (json, _) = source();
    let a = &json["accessors"][id];
    assert!(a.get("sparse").is_none());
    let view_id = a["bufferView"].as_u64().unwrap() as usize;
    let bytes = view(view_id);
    let width = match a["type"].as_str().unwrap() {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        _ => panic!("unsupported asset accessor"),
    };
    let component = a["componentType"].as_u64().unwrap();
    let size = if component == 5123 { 2 } else { 4 };
    let stride = json["bufferViews"][view_id]["byteStride"]
        .as_u64()
        .unwrap_or((width * size) as u64) as usize;
    let offset = a["byteOffset"].as_u64().unwrap_or(0) as usize;
    (0..a["count"].as_u64().unwrap() as usize)
        .map(|i| {
            (0..width)
                .map(|axis| {
                    let at = offset + i * stride + axis * size;
                    match component {
                        5126 => f32::from_bits(word(bytes, at)) as f64,
                        5125 => word(bytes, at) as f64,
                        5123 => u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()) as f64,
                        _ => panic!("unsupported asset component"),
                    }
                })
                .collect()
        })
        .collect()
}

fn vector(v: &Value, default: Vec3) -> Vec3 {
    v.as_array().map_or(default, |a| {
        Vec3::new(
            a[0].as_f64().unwrap(),
            a[1].as_f64().unwrap(),
            a[2].as_f64().unwrap(),
        )
    })
}

fn axes(p: Vec3) -> Vec3 {
    Vec3::new(-p.z, -p.x, p.y)
}

fn node(id: usize, linear: Mat3, translation: Vec3, inherited: Option<usize>, model: &mut Model) {
    let (json, _) = source();
    let n = &json["nodes"][id];
    assert!(n.get("matrix").is_none());
    let q = n["rotation"].as_array().map_or(Quat::IDENTITY, |v| {
        Quat::from_array(std::array::from_fn(|i| v[i].as_f64().unwrap()))
    });
    let translation = translation + linear * vector(&n["translation"], Vec3::ZERO);
    let linear = linear * Mat3::from_quat(q) * Mat3::from_diagonal(vector(&n["scale"], Vec3::ONE));
    let rotor = match n["name"].as_str().unwrap_or("") {
        "prop1" => Some(3),
        "prop2" => Some(1),
        "prop3" => Some(0),
        "prop4" => Some(2),
        _ => inherited,
    };
    if rotor != inherited {
        model.pivots[rotor.unwrap()] = axes(translation);
    }
    if let Some(mesh) = n["mesh"].as_u64() {
        for primitive in json["meshes"][mesh as usize]["primitives"]
            .as_array()
            .unwrap()
        {
            assert_eq!(primitive["mode"].as_u64().unwrap_or(4), 4);
            let values =
                |name: &str| accessor(primitive["attributes"][name].as_u64().unwrap() as usize);
            let normal_matrix = linear.inverse().transpose();
            let positions = values("POSITION")
                .iter()
                .map(|p| axes(translation + linear * Vec3::from_slice(p)))
                .collect();
            let normals = values("NORMAL")
                .iter()
                .map(|p| axes(normal_matrix * Vec3::from_slice(p)).normalize())
                .collect();
            let uv = values("TEXCOORD_0")
                .iter()
                .map(|p| [p[0] as f32, p[1] as f32])
                .collect();
            let indices = accessor(primitive["indices"].as_u64().unwrap() as usize)
                .iter()
                .map(|p| p[0] as usize)
                .collect();
            let material = &json["materials"][primitive["material"].as_u64().unwrap() as usize];
            let texture = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
                .as_u64()
                .unwrap() as usize;
            model.parts.push(Part {
                positions,
                normals,
                uv,
                indices,
                texture,
                rotor,
            });
        }
    }
    for child in n["children"].as_array().into_iter().flatten() {
        node(
            child.as_u64().unwrap() as usize,
            linear,
            translation,
            rotor,
            model,
        );
    }
}

pub fn model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| {
        let (json, _) = source();
        let mut model = Model {
            parts: Vec::new(),
            pivots: [Vec3::ZERO; 4],
        };
        for root in json["scenes"][0]["nodes"].as_array().unwrap() {
            node(
                root.as_u64().unwrap() as usize,
                Mat3::IDENTITY,
                Vec3::ZERO,
                None,
                &mut model,
            );
        }
        model
    })
}
