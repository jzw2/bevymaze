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

#import bevymaze::util lin_map

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
    var uv = in.uv;
    let r = sqrt(uv[0]*uv[0] + uv[1]*uv[1]);
    let theta = atan2(uv[1], uv[0]);
    // arsinh the magnitude and revert back to cart
    uv[0] = asinh(r*scale)/scale * cos(theta);
    uv[1] = asinh(r*scale)/scale * sin(theta);
    // Now do a linear transform to get to texture space
    uv[0] = lin_map(-u_bound, u_bound, 0.0, 1.0, uv[0]);
    uv[1] = lin_map(-v_bound, v_bound, 0.0, 1.0, uv[1]);
    // finally we can get the normal
    let n_vec4 = textureSample(normal_texture, normal_sampler, uv);
    // we have to remember that the normal is compressed!
    let x = n_vec4[0];
    let z = n_vec4[1];
    let normal = vec3<f32>(x, sqrt(1.0 - x*x - z*z), z);
//    let normal = in.world_normal;
//    pbr.world_normal = in.world_normal;
    pbr.world_normal = normal;
    pbr.N = pbr.world_normal;
    pbr.V = pbr_functions::calculate_view(in.world_position, false);

    var cosine_angle = dot(in.world_normal, vec3(0.0, 1.0, 0.0));

    var base_color = stone_color;
    let pos_vector = vec2<f32>(in.original_world_position[0] * 0.001, in.original_world_position[2] * 0.001);
    let height_frac = in.original_world_position[1] / max_height + 0.0; //0.08 * perlin_noise_2d(pos_vector);

    pbr.material.perceptual_roughness = 0.98;
    pbr.material.reflectance = 0.001;

    if (height_frac < grass_line) {
        base_color = grass_color;
    } else if (height_frac < tree_line) {
        if (cosine_angle > cosine_max_tree_slope) {
            base_color = tree_color;
        } else {
            base_color = grass_color;
        }
    } else if (height_frac > snow_line && cosine_angle > cosine_max_snow_slope) {
        pbr.material.reflectance = 0.95;
        pbr.material.perceptual_roughness = 0.2;
        base_color = snow_color;
    }

    pbr.material.base_color = base_color;

    var output_color = pbr_functions::apply_pbr_lighting(pbr);
    output_color = pbr_functions::apply_fog(fog, output_color, in.world_position.xyz, view.world_position.xyz);

    return output_color;
}
