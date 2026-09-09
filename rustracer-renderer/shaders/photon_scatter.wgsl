// Photon grid build pass 2: scatter full photon structs into sorted_photons by cell.

#include common.wgsl

@group(0) @binding(0) var<storage, read> photons: array<Photon>;
@group(0) @binding(1) var<storage, read> grid_meta: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> grid_cursor: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sorted_photons: array<Photon>;
@group(0) @binding(4) var<uniform> gp: GridParams;

struct GridParams {
    room_min_x: f32, room_min_y: f32, room_min_z: f32,
    cell_size: f32,
    grid_x: u32, grid_y: u32, grid_z: u32,
    num_cells: u32,
    photon_slots: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
}

fn photon_cell(pos: vec3<f32>) -> u32 {
    let gx = u32((pos.x - gp.room_min_x) / gp.cell_size);
    let gy = u32((pos.y - gp.room_min_y) / gp.cell_size);
    let gz = u32((pos.z - gp.room_min_z) / gp.cell_size);
    return gx + gy * gp.grid_x + gz * gp.grid_x * gp.grid_y;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slot = global_id.x;
    if slot >= gp.photon_slots { return; }
    let ph = photons[slot];
    if ph.power.r <= 0.0 { return; }
    let cell = photon_cell(ph.position);
    if cell >= gp.num_cells { return; }
    let pos = atomicAdd(&grid_cursor[cell], 1u);
    sorted_photons[grid_meta[cell].x + pos] = ph;
}
