#define_import_path bevymaze::curvature_mesh_vertex_output

struct CuravtureMeshVertexOutput {
    // this is `clip position` when the struct is used as a vertex stage output
    // and `frag coord` when used as a fragment stage input
    @builtin(position) position: vec4<f32>,
    @location(0) original_world_position: vec4<f32>,
    @location(1) world_position: vec4<f32>,

    #ifdef VERTEX_NORMALS
    @location(2) world_normal: vec3<f32>,
    #endif

    #ifdef VERTEX_UVS
    @location(3) uv: vec2<f32>,
    #endif

    #ifdef VERTEX_TANGENTS
    @location(4) world_tangent: vec4<f32>,
    #endif

    #ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
    #endif

    #ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    @location(6) instance_index: u32,
    #endif
}