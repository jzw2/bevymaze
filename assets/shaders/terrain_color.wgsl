#define_import_path bevy_pbr::fragment

#import bevy_pbr::pbr_functions as pbr_functions
#import bevy_pbr::pbr_bindings as pbr_bindings
#import bevy_pbr::pbr_types as pbr_types
#import bevy_pbr::prepass_utils

//#import bevy_pbr::mesh_vertex_output       MeshVertexOutput
#import bevymaze::curvature_mesh_vertex_output                 CuravtureMeshVertexOutput
#import bevy_pbr::mesh_bindings            mesh
#import bevy_pbr::mesh_view_bindings       view, fog, screen_space_ambient_occlusion_texture
#import bevy_pbr::mesh_view_types          FOG_MODE_OFF
#import bevy_core_pipeline::tonemapping    screen_space_dither, powsafe, tone_mapping
#import bevy_pbr::parallax_mapping         parallaxed_uv

#import bevy_pbr::prepass_utils

#import bevy_pbr::gtao_utils gtao_multibounce

//#import bevy_shader_utils::perlin_noise_2d perlin_noise_2d

#import bevymaze::util::{lin_map, hash12}

#import bevy_pbr::{
    //forward_io::VertexOutput,
    //mesh_view_bindings::view,
    pbr_types::{STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT, PbrInput, pbr_input_new},
    //pbr_functions as fns,
}

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
var<uniform> u_bound: f32;
@group(1) @binding(11)
var<uniform> v_bound: f32;
@group(1) @binding(12)
var normal_texture: texture_2d<f32>;
@group(1) @binding(13)
var normal_sampler: sampler;
@group(1) @binding(14)
var<uniform> scale: f32;

@fragment
fn fragment(
    in: CuravtureMeshVertexOutput,
) -> @location(0) vec4<f32> {
    var pbr = pbr_input_new();
    pbr.frag_coord = in.position;
    pbr.world_position = in.world_position;

    // The UV is simply the x/z components of our vert
    // Take this and map to the unnormalized position in ellipse space
    // Do the polar arsinh transform
    // Get the polar
    var uv = in.world_position.xz;
    let length = length(uv);
    let scale = (asinh(length * scale) / scale) / length;
    uv *= scale;

    // Now do a linear transform to get to texture space
    uv.x = lin_map(-u_bound, u_bound, 0.0, 1.0, uv.x);
    uv.y = lin_map(-v_bound, v_bound, 0.0, 1.0, uv.y);
    // finally we can get the normal
    let n_vec4 = textureSample(normal_texture, normal_sampler, uv);
    // we have to remember that the normal is compressed!
    let x = n_vec4.x;
    let z = n_vec4.y;
    let normal = vec3<f32>(x, sqrt(1.0 - x*x - z*z), z);
    pbr.world_normal = normal;
    pbr.N = pbr.world_normal;
    pbr.V = pbr_functions::calculate_view(in.world_position, false);

    var cosine_angle = dot(pbr.world_normal, vec3(0.0, 1.0, 0.0));

    var base_color = stone_color;
    let height_frac = in.original_world_position[1] / max_height;

    pbr.material.perceptual_roughness = 0.98;
    pbr.material.reflectance = 0.2;

    if (height_frac < grass_line) {
        base_color = grass_color * 0.5;
    } else if (height_frac < tree_line) {
        if (cosine_angle > cosine_max_tree_slope) {
            base_color = tree_color * 0.5;
        } else {
            base_color = grass_color * 0.5;
        }
    } else if (height_frac > snow_line && cosine_angle > cosine_max_snow_slope) {
        let flicker = floor(in.world_position.xz * 20.0f);
        let rand = hash12(flicker);
        pbr.material.reflectance = 0.95;
//        pbr.material.perceptual_roughness = 0.01;
        base_color = snow_color;
        if rand > 0.5 {
            base_color -= 0.6 * vec4<f32>(1.0, 1.0, 1.0, 0.0);
        }
    }

    pbr.material.base_color = base_color;

    var output_color = pbr_functions::apply_pbr_lighting(pbr);
    output_color = pbr_functions::apply_fog(fog, output_color, in.world_position.xyz, view.world_position.xyz);

    return output_color;
}
