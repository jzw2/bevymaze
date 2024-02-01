///// Raw triples (xyz) of data unorganized terrain data.
//@group(0) @binding(0) var<storage, read> terrain_data_verts: array<f32>;
///// A list of triples of indices representing the Delaunay triangulation.
///// Each number corresponds to the index of a datapoint in terrain_data_verts
//@group(0) @binding(1) var<storage, read> triangulation: array<u32>;
///// A list of triples of points representing the xy and z of the circumcenters
///// Such that doing the nearest neighbor search in the xy plane solves
///// the point location problem.
///// The circumcenter at 3*i corresponds to the triangle at 3*i
//@group(0) @binding(2) var<storage, read> circumcenters: array<u32>;
//
///// The vertices of the actual mesh
//@group(0) @binding(3) var<storage, read> mesh_verts: array<f32>;

@group(0) @binding(999) var<uniform, read_write> intensity: f32;

/// Coordinates representing the nearest triangle
struct NearestTriangleCoords {
    /// Index of the vertices in the triangulation
    idxA: u32,
    idxB: u32,
    idxC: u32,
    /// Barycentric coordinates. A,B,C coorespond the verts at idx+0,idx+1,idx+2 respectively
    bA: f32,
    bB: f32,
    bC: f32,
}

/// The calculated triangle and barycentric coords representing the triangle
/// that a vertex falls in
/// Coords at i correspond to a vert at 3*i
//@group(0) @binding(4) var<storage, write> triangle_coords: array<NearestTriangleCoords>;

@compute @workgroup_size(256, 1, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    /// Do some fancy stuff
    intensity = 2.0;
}