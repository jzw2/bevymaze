#define_import_path bevy_pbr::fragment

#import bevy_pbr::pbr_functions as pbr_functions
#import bevy_pbr::pbr_bindings as pbr_bindings
#import bevy_pbr::pbr_types as pbr_types
#import bevy_pbr::prepass_utils

#import bevy_pbr::mesh_vertex_output       MeshVertexOutput
#import bevy_pbr::mesh_bindings            mesh
#import bevy_pbr::mesh_view_bindings       view, fog, screen_space_ambient_occlusion_texture
#import bevy_pbr::mesh_view_types          FOG_MODE_OFF
#import bevy_core_pipeline::tonemapping    screen_space_dither, powsafe, tone_mapping
#import bevy_pbr::parallax_mapping         parallaxed_uv

#import bevy_pbr::prepass_utils

#import bevy_pbr::gtao_utils gtao_multibounce

#import bevy_shader_utils::perlin_noise_2d perlin_noise_2d

@group(1) @binding(0)
var<uniform> max_height: f32;
@group(1) @binding(1)
var<uniform> grass_line: f32;
@group(1) @binding(2)
var<uniform> tree_line: f32;
@group(1) @binding(3)
var<uniform> snow_line: f32;
@group(1) @binding(4)
var<uniform> grass_color: vec4<f32>;
@group(1) @binding(5)
var<uniform> tree_color: vec4<f32>;
@group(1) @binding(6)
var<uniform> snow_color: vec4<f32>;
@group(1) @binding(7)
var<uniform> stone_color: vec4<f32>;
@group(1) @binding(8)
var<uniform> cosine_max_snow_slope: f32;
@group(1) @binding(9)
var<uniform> cosine_max_tree_slope: f32;
@group(1) @binding(10)
var normal_texture: texture_2d<f32>;
@group(1) @binding(11)
var normal_sampler: sampler;

@fragment
fn fragment(
    in: MeshVertexOutput,
) -> @location(0) vec4<f32> {
    var pbr = pbr_functions::pbr_input_new();
    pbr.frag_coord = in.position;
    pbr.world_position = in.world_position;

    let n_vec4 = textureSample(normal_texture, normal_sampler, in.uv);
    let normal = vec3(n_vec4[0], n_vec4[1], n_vec4[2]);
    pbr.world_normal = normal;
    pbr.N = normal;
    pbr.V = pbr_functions::calculate_view(in.world_position, false);

    var cosine_angle = dot(in.world_normal, vec3(0.0, 1.0, 0.0));

    var base_color = stone_color;
    let pos_vector = vec2<f32>(in.world_position[0] * 0.001, in.world_position[2] * 0.001);
    let height_frac = in.world_position[1] / max_height + 0.08 * perlin_noise_2d(pos_vector);
    if (height_frac < grass_line) {
        base_color = grass_color;
    } else if (height_frac < tree_line) {
        if (cosine_angle > cosine_max_tree_slope) {
            base_color = tree_color;
        } else {
            base_color = grass_color;
        }
    } else if (height_frac > snow_line && cosine_angle > cosine_max_snow_slope) {
        base_color = snow_color;
    }

    pbr.material.base_color = base_color;
    pbr.material.perceptual_roughness = 0.95;
    pbr.material.reflectance = 0.01;
    var output_color = pbr_functions::pbr(pbr);

    return output_color;
}