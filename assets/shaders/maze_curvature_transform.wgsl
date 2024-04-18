#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::mesh_bindings       mesh
#import bevy_pbr::mesh_view_bindings
//#import bevy_pbr::mesh_vertex_output  MeshVertexOutput
#import bevymaze::util::{lin_map, hash11, murmurHash11}
#import bevymaze::curvature_mesh_vertex_output  CuravtureMeshVertexOutput

#import bevy_pbr::mesh_functions
#import bevy_pbr::skinning
#import bevy_pbr::morph
#import bevy_pbr::{
    //mesh_view_bindings::view,
    //mesh_bindings::mesh,
    //mesh_types::MESH_FLAGS_SIGN_DETERMINANT_MODEL_3X3_BIT,
    view_transformations::position_world_to_clip,
}

//#import bevy_render::instance_index

/// COPIED FROM instance_index.wgsl

#ifdef BASE_INSTANCE_WORKAROUND
// naga and wgpu should polyfill WGSL instance_index functionality where it is
// not available in GLSL. Until that is done, we can work around it in bevy
// using a push constant which is converted to a uniform by naga and wgpu.
// https://github.com/gfx-rs/wgpu/issues/1573
var<push_constant> base_instance: i32;

fn get_instance_index(instance_index: u32) -> u32 {
    return u32(base_instance) + instance_index;
}
#else
fn get_instance_index(instance_index: u32) -> u32 {
    return instance_index;
}
#endif

/// END COPIED SECTION


struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
#ifdef VERTEX_POSITIONS
    @location(0) position: vec3<f32>,
#endif
#ifdef VERTEX_NORMALS
    @location(1) normal: vec3<f32>,
#endif
#ifdef VERTEX_UVS
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_TANGENTS
    @location(3) tangent: vec4<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(4) color: vec4<f32>,
#endif
};

const max_u32: u32 = 4294967295u;

@group(1) @binding(10)
var<uniform> u_bound: f32;
@group(1) @binding(11)
var<uniform> v_bound: f32;

/// len is MAX_TRIANGLES * 3
@group(1) @binding(15)
var<storage, read> triangles: array<u32>;

/// len is MAX_TRIANGLES * 3
@group(1) @binding(16)
var<storage, read> halfedges: array<u32>;

/// len is 2 * MAX_VERTICES
@group(1) @binding(17)
var<storage, read> vertices: array<f32>;

/// len is MAX_VERTICES
@group(1) @binding(18)
var<storage, read> height: array<f32>;

/// len is 2 * MAX_VERTICES
@group(1) @binding(19)
var<storage, read> gradients: array<f32>;

/// len is TERRAIN_VERTICES
@group(1) @binding(20)
var<storage, read_write> triangle_indices: array<u32>;

/// len is TERRAIN_VERTICES
@group(1) @binding(23)
var<uniform> layer_height: f32;

// Copied and adapted from https://gamedev.stackexchange.com/a/23745
// Compute barycentric coordinates (u, v, w) for
// point p with respect to triangle (a, b, c)
fn barycentric(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec3<f32> {
    let v0 = b - a; let v1 = c - a; let v2 = p - a;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    return vec3<f32>(v, w, 1.0 - v - w);
}

fn interp(ai: u32, bi: u32, ci: u32, bary: vec2<f32>) -> f32 {
    let x: f32 = bary.x;
    let y: f32 = bary.y;

    let p10: f32 = height[ai];
    let p9 = gradients[ai * 2u + 1u];
    let p8 = gradients[ai * 2u];
    // let p7 = f32(0); // ignore, but here because it looks nicer
    let p2 = -2.0 * height[ci] + p9 + 2.0 * p10 + gradients[ci * 2u + 1u];
    let p6 = height[ci] - p9 - p10 - p2;
    let p1 = -2.0 * height[bi] + p8 + 2.0 * p10 + gradients[bi * 2u];
    let p5 = height[bi] - p8 - p10 - p1;
    let p4 = gradients[ci * 2u] - p8;
    let p3 = gradients[bi * 2u + 1u] - p9;

    return
        p1 * x*x*x + p2 * y*y*y + p3 * x*x * y + p4 * x * y*y +
        p5 * x*x + p6 * y*y + // p7 * x * y
        p8 * x + p9 * y +
        p10;
}

/// I just copy and pasted this stuff because i'm whacky
@vertex
fn vertex(vertex_no_morph: Vertex) -> CuravtureMeshVertexOutput {
    var out: CuravtureMeshVertexOutput;

    var vertex = vertex_no_morph;

    // Use vertex_no_morph.instance_index instead of vertex.instance_index to work around a wgpu dx12 bug.
    // See https://github.com/gfx-rs/naga/issues/2416 .
    var model = mesh_functions::get_model_matrix(vertex_no_morph.instance_index);

#ifdef VERTEX_NORMALS
    out.world_normal = bevy_pbr::mesh_functions::mesh_normal_local_to_world(
        vertex.normal,
//        // Use vertex_no_morph.instance_index instead of vertex.instance_index to work around a wgpu dx12 bug.
//        // See https://github.com/gfx-rs/naga/issues/2416
        get_instance_index(vertex_no_morph.instance_index)
    );
#endif

#ifdef VERTEX_POSITIONS

    out.uv = vertex.uv;
    out.world_position = bevy_pbr::mesh_functions::mesh_position_local_to_world(model, vec4<f32>(vertex.position, 1.0));

    let found = triangle_indices[vertex.vertex_index];
    if found != max_u32 {
        let tri = 3u * (found / 3u);
        let ai = 2u * triangles[tri];
        let bi = 2u * triangles[tri + 1u];
        let ci = 2u * triangles[tri + 2u];
        let a = vec2<f32>(vertices[ai], vertices[ai + 1u]);
        let b = vec2<f32>(vertices[bi], vertices[bi + 1u]);
        let c = vec2<f32>(vertices[ci], vertices[ci + 1u]);
        let bary = barycentric(out.world_position.xz, a, b, c);
        let height_hash = hash11(layer_height);
        let height_hash2 = hash11(layer_height + 1.0);
        let height_hash3 = hash11(layer_height + 2.0);
        let perturbance = sin(4.0 * (height_hash * out.world_position.xz + 2.0) + vec2<f32>(height_hash2, height_hash3));
        out.world_position.y = interp(ai / 2u, bi / 2u, ci / 2u, bary.xy)
            + layer_height
            + 0.35 * (perturbance.x + perturbance.y) / 2.0;
    } else {
        out.world_position.y = 0.0;
    }

    // original being before the curvature, not the height in this case
    out.original_world_position = out.world_position;
    /// Here we calculate the height that we need to subtract
    /// this is dependent on the distance from the origin
    /// The process is to get the "flat" offset (along the x-axis) and then rotate it back to its original position
    /// The offset is simply
    /// ( r*sin(theta), r*cos(theta), 0)
    /// where r = r_e + y, theta = sqrt(v_x^2 + v_z^2) / r_e
    /// where r_e is the radius of the earth and v is the vertex's world pos
    let view_world_pos = bevy_pbr::mesh_view_bindings::view.world_position.xyz;
    let rel_pos = out.world_position.xyz - view_world_pos;
    let origin_dist: f32 = sqrt(rel_pos.x*rel_pos.x + rel_pos.z*rel_pos.z);
    if origin_dist > 1000.0 {
        // only apply our transformation at large distances, because otherwise
        // rounding errors become noticable
        let EARTH_RAD: f32 = (6.378e+6);
        let theta: f32 = origin_dist / EARTH_RAD;
        let r = EARTH_RAD + out.world_position.y;
        let unrotated_pos = vec3<f32>(r * sin(theta), r * (cos(theta) - 1.0), 0.0);
        /// Now we rotate it to match the original
        /// (signed) angle of the original is phi = atan2(v_z, v_x * v_z)
        var phi: f32 = 0.0;
        if (rel_pos[0] != 0.0) {
            phi = -atan2(rel_pos.z, rel_pos.x);
        }
        let rotation_mat = mat3x3<f32>(cos(phi), 0.0, -sin(phi), 0.0, 1.0, 0.0, sin(phi), 0.0, cos(phi));
        let mapped: vec3<f32> = rotation_mat*unrotated_pos + view_world_pos;
        out.world_position =
            vec4<f32>(
                mapped.x,
                mapped.y + out.world_position.y - view_world_pos.y,
                mapped.z,
                out.world_position.w);
    }
    out.position = position_world_to_clip(out.world_position.xyz);
#endif

#ifdef VERTEX_UVS
    out.uv = vertex.uv;
#endif

#ifdef VERTEX_TANGENTS
    out.world_tangent = bevy_pbr::mesh_functions::mesh_tangent_local_to_world(
        model,
        vertex.tangent,
//        // Use vertex_no_morph.instance_index instead of vertex.instance_index to work around a wgpu dx12 bug.
//        // See https://github.com/gfx-rs/naga/issues/2416
//        get_instance_index(vertex_no_morph.instance_index)
    );
#endif

#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    // Use vertex_no_morph.instance_index instead of vertex.instance_index to work around a wgpu dx12 bug.
    // See https://github.com/gfx-rs/naga/issues/2416
    out.instance_index = get_instance_index(vertex_no_morph.instance_index);
#endif

    return out;
}