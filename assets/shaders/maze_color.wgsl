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

#import bevymaze::util::{lin_map, hash12, hash11, positive_rem}

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

@group(1) @binding(23)
var<uniform> layer_height: f32;

const MAZE_DATA_COUNT = 16;
const MAZE_CELLS_X = 4;
const MAZE_CELLS_Y = 4;

//const MAZE_CELL_SIZE = 128.0;

@group(1) @binding(24)
var<storage, read> maze: array<array<array<u32, MAZE_DATA_COUNT>, MAZE_CELLS_Y>, MAZE_CELLS_X>;

@group(1) @binding(25)
var<uniform> maze_top_left: vec2<f32>;

fn is_in_path(pos: vec2<f32>) -> bool {
    let top_left_i32 = vec2<i32>(i32(maze_top_left.x), i32(maze_top_left.y));
    var i = i32(floor(pos.x / 64.0f));
    var j = i32(floor(pos.y / 64.0f));
    i = i - top_left_i32.x;
    j = j - top_left_i32.y;
    i = clamp(i, 0, MAZE_CELLS_X - 1);
    j = clamp(j, 0, MAZE_CELLS_X - 1);

    let cell = vec2<u32>(u32(floor(positive_rem(pos.x, 64.0f))), u32(floor(positive_rem(pos.y, 64.0f))));
    let quad = vec2<u32>(u32(floor(positive_rem(pos.x, 2.0f))),  u32(floor(positive_rem(pos.y, 2.0f))));

    if quad.x == 0u && quad.y == 0u {
        return false;
    } else if quad.x == 1u && quad.y == 1u {
        return true;
    } else if quad.x == 0u && quad.y == 1u {
        let bit = u32((cell.x + cell.y * 16u) * 2u);
        let word = u32(bit / 32u);
        let offset = u32(bit % 32u);
        let word_val = u32(maze[i][j][word]);
        return ((word_val >> offset) & 1u) == 1u;
    } else if quad.x == 1u && quad.y == 0u {
        let bit = u32((cell.x + cell.y * 16u) * 2u) + 1u;
        let word = bit / 32u;
        let offset = u32(bit % 32u);
        let word_val = u32(maze[i][j][word]);
        return ((word_val >> offset) & 1u) == 1u;
    }
    return true;
}

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
    pbr.world_normal = normal;
    pbr.N = pbr.world_normal;
    pbr.V = pbr_functions::calculate_view(in.world_position, false);

    let height_frac = in.original_world_position[1] / max_height + 0.0; //0.08 * perlin_noise_2d(pos_vector);
    if (height_frac >= grass_line) {
        discard;
    }

    let leaf = floor(in.world_position.xz * 10.0f);

    if is_in_path(in.world_position.xz) {
        discard;
    }

    let rand = hash12(leaf);
    let norm_l_height = layer_height / 3.0f;
    if rand > hash11(layer_height) * 0.9 {
        pbr.material.base_color = grass_color * norm_l_height;
    } else {
        discard;
    }

    pbr.material.perceptual_roughness = 0.98;
    pbr.material.reflectance = 0.001;
    var output_color = pbr_functions::apply_pbr_lighting(pbr);
    output_color = pbr_functions::apply_fog(fog, output_color, in.world_position.xyz, view.world_position.xyz);

    return output_color;
}
