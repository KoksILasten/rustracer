use glam::Vec3;
use rustracer_core::loader::load_gltf;
use serde_json::{Value, json};
use std::{path::PathBuf, sync::atomic::{AtomicUsize, Ordering}};

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

struct Fixture { dir: PathBuf, path: PathBuf }
impl Fixture {
    fn new(mut doc: Value, bin: Vec<u8>, glb: bool) -> Self {
        let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("rustracer-gltf-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        doc["buffers"] = json!([{"byteLength": bin.len()}]);
        let path = dir.join(if glb { "scene.glb" } else { "scene.gltf" });
        if glb {
            let mut json = serde_json::to_vec(&doc).unwrap();
            while json.len() % 4 != 0 { json.push(b' '); }
            let mut bin = bin;
            while bin.len() % 4 != 0 { bin.push(0); }
            let mut bytes = Vec::new();
            for word in [0x46546c67u32, 2, (12 + 8 + json.len() + 8 + bin.len()) as u32,
                json.len() as u32, 0x4e4f534a] { bytes.extend(word.to_le_bytes()); }
            bytes.extend(json);
            bytes.extend((bin.len() as u32).to_le_bytes());
            bytes.extend(0x004e4942u32.to_le_bytes());
            bytes.extend(bin);
            std::fs::write(&path, bytes).unwrap();
        } else {
            doc["buffers"][0]["uri"] = json!("geometry.bin");
            std::fs::write(dir.join("geometry.bin"), bin).unwrap();
            std::fs::write(&path, serde_json::to_vec(&doc).unwrap()).unwrap();
        }
        Self { dir, path }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.dir); }
}

fn geometry(positions: &[[f32; 3]], normals: Option<&[[f32; 3]]>, indices: Option<&[u32]>, mode: u32) -> (Value, Vec<u8>) {
    let mut bin = Vec::new();
    let mut views = Vec::new();
    let mut accessors = Vec::new();
    for vectors in std::iter::once(positions).chain(normals) {
        let offset = bin.len();
        for v in vectors { for f in v { bin.extend(f.to_le_bytes()); } }
        views.push(json!({"buffer": 0, "byteOffset": offset, "byteLength": bin.len()-offset}));
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for v in vectors { for i in 0..3 { min[i] = min[i].min(v[i]); max[i] = max[i].max(v[i]); } }
        accessors.push(json!({"bufferView": views.len()-1, "componentType":5126, "count": vectors.len(), "type":"VEC3", "min":min, "max":max}));
    }
    let mut primitive = json!({"attributes":{"POSITION":0}, "mode":mode});
    if normals.is_some() { primitive["attributes"]["NORMAL"] = json!(1); }
    if let Some(indices) = indices {
        let offset = bin.len();
        for i in indices { bin.extend(i.to_le_bytes()); }
        views.push(json!({"buffer":0,"byteOffset":offset,"byteLength":bin.len()-offset}));
        accessors.push(json!({"bufferView":views.len()-1,"componentType":5125,"count":indices.len(),"type":"SCALAR"}));
        primitive["indices"] = json!(accessors.len()-1);
    }
    (json!({"asset":{"version":"2.0"},"bufferViews":views,"accessors":accessors,
        "meshes":[{"primitives":[primitive]}],"nodes":[{"mesh":0}],"scenes":[{"nodes":[0]}],"scene":0}), bin)
}
fn triangle() -> (Value, Vec<u8>) {
    geometry(&[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]], Some(&[[0.,0.,1.];3]), None, 4)
}
fn near(actual: Vec3, expected: Vec3) {
    assert!((actual-expected).length() < 1e-5, "actual {actual:?}, expected {expected:?}");
}

#[test]
fn default_scene_traverses_hierarchy_and_instances_only_referenced_meshes() {
    let (mut doc, bin) = triangle();
    let mesh0 = doc["meshes"][0].clone();
    doc["meshes"].as_array_mut().unwrap().push(mesh0);
    doc["nodes"] = json!([
        {"mesh":1,"translation":[100,0,0]},
        {"translation":[10,0,0],"children":[2,3]},
        {"mesh":0,"translation":[0,2,0],"scale":[2,3,1]},
        {"mesh":0,"translation":[0,0,5]}
    ]);
    doc["scenes"] = json!([{"nodes":[0]},{"nodes":[1]}]);
    doc["scene"] = json!(1);
    for glb in [false, true] {
        let fixture = Fixture::new(doc.clone(), bin.clone(), glb);
        let scene = load_gltf(&fixture.path).unwrap();
        assert_eq!(scene.meshes.len(), 2);
        near(scene.meshes[0].positions[1], Vec3::new(12.,2.,0.));
        near(scene.meshes[1].positions[1], Vec3::new(11.,0.,5.));
    }
}
