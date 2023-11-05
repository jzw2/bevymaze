#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::mesh_bindings       mesh
#import bevy_pbr::mesh_view_bindings
//#import bevy_pbr::mesh_vertex_output  MeshVertexOutput
#import bevymaze::curvature_mesh_vertex_output  CuravtureMeshVertexOutput

#import bevy_pbr::mesh_functions
#import bevy_pbr::skinning
#import bevy_pbr::morph

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

/// I just copy and pasted this stuff because i'm whacky
@vertex
fn vertex(vertex_no_morph: Vertex) -> CuravtureMeshVertexOutput {
    var out: CuravtureMeshVertexOutput;

    var vertex = vertex_no_morph;

    // Use vertex_no_morph.instance_index instead of vertex.instance_index to work around a wgpu dx12 bug.
    // See https://github.com/gfx-rs/naga/issues/2416 .
    var model = mesh.model;

#ifdef VERTEX_NORMALS
    out.world_normal = bevy_pbr::mesh_functions::mesh_normal_local_to_world(
        vertex.normal,
//        // Use vertex_no_morph.instance_index instead of vertex.instance_index to work around a wgpu dx12 bug.
//        // See https://github.com/gfx-rs/naga/issues/2416
//        get_instance_index(vertex_no_morph.instance_index)
    );
#endif

#ifdef VERTEX_POSITIONS
    out.world_position = bevy_pbr::mesh_functions::mesh_position_local_to_world(model, vec4<f32>(vertex.position, 1.0));
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
    let origin_dist: f32 = sqrt(rel_pos[0]*rel_pos[0] + rel_pos[2]*rel_pos[2]);
    if origin_dist > 1000.0 {
        // only apply our transformation at large distances, because otherwise
        // rounding errors become noticable
        let EARTH_RAD: f32 = (6.378e+6)/4.0;
        let theta: f32 = origin_dist / EARTH_RAD;
        let r = EARTH_RAD + out.world_position[1];
        let unrotated_pos = vec3<f32>(r * sin(theta), r * (cos(theta) - 1.0), 0.0);
        /// Now we rotate it to match the original
        /// (signed) angle of the original is phi = atan2(v_z, v_x * v_z)
        var phi: f32 = 0.0;
        if (rel_pos[0] != 0.0) {
            phi = -atan2(rel_pos[2], rel_pos[0]);
        }
        let rotation_mat = mat3x3<f32>(cos(phi), 0.0, -sin(phi), 0.0, 1.0, 0.0, sin(phi), 0.0, cos(phi));
        let mapped: vec3<f32> = rotation_mat*unrotated_pos + view_world_pos;
        out.world_position = vec4<f32>(mapped[0], mapped[1] + out.world_position[1] - view_world_pos[1], mapped[2], out.world_position[3]);
    }
    out.position = bevy_pbr::mesh_functions::mesh_position_world_to_clip(out.world_position);
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


//#import bevy_pbr::mesh_view_bindings
//#import bevy_pbr::mesh_bindings
//
//// NOTE: Bindings must come before functions that use them!
//#import bevy_pbr::mesh_functions
//
//#import bevy_shader_utils::simplex_noise_3d
//#import bevy_shader_utils::simplex_noise_2d
//
//struct StandardMaterial {
//    time: f32,
//    // ship_position: vec3<f32>,
//    // base_color: vec4<f32>;
//    // emissive: vec4<f32>;
//    // perceptual_roughness: f32;
//    // metallic: f32;
//    // reflectance: f32;
//    // // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
//    // flags: u32;
//    // alpha_cutoff: f32;
//};
//
//@vertex
//fn vertex(vertex: Vertex) -> MeshVertexOutput {
//    var out: MeshVertexOutput;
//
//    var model = mesh.model;
//    out.world_normal = mesh_normal_local_to_world(vertex.normal);
//
//    out.world_position = mesh_position_local_to_world(model, vec4<f32>(vertex.position, 1.0));
//#ifdef VERTEX_UVS
//    out.uv = vertex.uv;
//#endif
//#ifdef VERTEX_TANGENTS
//    out.world_tangent = mesh_tangent_local_to_world(model, vertex.tangent);
//#endif
//#ifdef VERTEX_COLORS
//    out.color = vertex.color;
//#endif
//
//
//    var noise = simplexNoise3(vertex.position);
//
//    out.color = vec4<f32>(0.5, noise, 0.5, 1.0);
//
//    out.world_position = mesh_position_local_to_world(model, vec4<f32>(vertex.position.x, noise, vertex.position.z, 1.0));
//
//    out.clip_position = mesh_position_world_to_clip(out.world_position);
//
//
//    return out;
//}