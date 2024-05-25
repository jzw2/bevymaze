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

#import bevymaze::util::{lin_map, hash12, hash11, positive_rem, pos_rem_vec}

#import bevy_pbr::{
    //forward_io::VertexOutput,
    //mesh_view_bindings::view,
    pbr_types::{STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT, PbrInput, pbr_input_new},
    //pbr_functions as fns,
}

@group(2) @binding(0)
var<uniform> max_height: f32;
@group(2) @binding(1)
var<uniform> grass_line: f32;
@group(2) @binding(2)
var<uniform> tree_line: f32;
@group(2) @binding(3)
var<uniform> snow_line: f32;
@group(2) @binding(4)
var<uniform> grass_color: vec4<f32>;
@group(2) @binding(5)
var<uniform> tree_color: vec4<f32>;
@group(2) @binding(6)
var<uniform> snow_color: vec4<f32>;
@group(2) @binding(7)
var<uniform> stone_color: vec4<f32>;
@group(2) @binding(8)
var<uniform> cosine_max_snow_slope: f32;
@group(2) @binding(9)
var<uniform> cosine_max_tree_slope: f32;
@group(2) @binding(10)
var<uniform> u_bound: f32;
@group(2) @binding(11)
var<uniform> v_bound: f32;
@group(2) @binding(12)
var normal_texture: texture_2d<f32>;
@group(2) @binding(13)
var normal_sampler: sampler;
@group(2) @binding(14)
var<uniform> scale: f32;

@group(2) @binding(23)
var<uniform> layer_height: f32;

const COMPONENT_CELLS: u32 = #{COMPONENT_CELLS_DEF__};
const PATH_WIDTH: f32 = f32(#{PATH_WIDTH_DEF__});
const MAZE_DATA_COUNT: u32 = COMPONENT_CELLS * COMPONENT_CELLS * 2u / 32u;
const MAZE_COMPONENTS = #{MAZE_COMPONENTS_DEF__};
const COMPONENT_SIZE: f32 = PATH_WIDTH * 2.0 * f32(COMPONENT_CELLS);

//const MAZE_COMPONENT_SIZE = 128.0;

@group(2) @binding(24)
var<storage, read> maze: array<array<array<u32, MAZE_DATA_COUNT>, MAZE_COMPONENTS>, MAZE_COMPONENTS>;

@group(2) @binding(25)
var<uniform> maze_top_left: vec2<f32>;

fn is_in_path(pos: vec2<f32>) -> bool {
    let top_left_i32 = vec2<i32>(i32(maze_top_left.x), i32(maze_top_left.y));
    var comp = vec2<i32>(floor(pos / COMPONENT_SIZE));
    comp -= top_left_i32;
    comp.x = clamp(comp.x, 0, i32(MAZE_COMPONENTS) - 1);
    comp.y = clamp(comp.y, 0, i32(MAZE_COMPONENTS) - 1);

    let cell = vec2<u32>(floor(pos_rem_vec(pos / (PATH_WIDTH * 2.0), COMPONENT_SIZE)));
    let quad = pos_rem_vec(pos, PATH_WIDTH * 2.0);

    if quad.x < PATH_WIDTH && quad.y < PATH_WIDTH {
        return false;
    } else if quad.x >= PATH_WIDTH && quad.y >= PATH_WIDTH {
        return true;
    } else if quad.x < PATH_WIDTH && quad.y >= PATH_WIDTH {
        let bit = u32((cell.x + cell.y * COMPONENT_CELLS) * 2u);
        let word = bit / 32u;
        let offset = bit % 32u;
        let word_val = maze[comp.x][comp.y][word];
        return ((word_val >> offset) & 1u) == 1u;
    } else if quad.x >= PATH_WIDTH && quad.y < PATH_WIDTH {
        let bit = u32((cell.x + cell.y * COMPONENT_CELLS) * 2u) + 1u;
        let word = bit / 32u;
        let offset = bit % 32u;
        let word_val = maze[comp.x][comp.y][word];
        return ((word_val >> offset) & 1u) == 1u;
    }
    return true;
}

fn sqr_dist_to_unit_square(pos: vec2<f32>, square_pos: vec2<f32>) -> f32 {
    let r = floor(square_pos / PATH_WIDTH) * PATH_WIDTH + PATH_WIDTH / 2.0;
    let d = max(abs(pos - r) - PATH_WIDTH / 2.0, vec2<f32>(0.0, 0.0));
    return d.x * d.x + d.y * d.y;
}

fn dist_to_path(pos: vec2<f32>) -> f32 {
    if is_in_path(pos) {
        return 0.0;
    }

    var sqr_dist = PATH_WIDTH * PATH_WIDTH;
    for(var x: i32 = -1; x <= 1; x++) {
        for(var y: i32 = -1; y <= 1; y++) {
            if x == 0 && y == 0 {
                continue;
            }
            let square_pos = pos + vec2<f32>(PATH_WIDTH * f32(x), PATH_WIDTH * f32(y));
            if is_in_path(square_pos) {
                sqr_dist = min(sqr_dist, sqr_dist_to_unit_square(pos, square_pos));
            }
        }
    }

    return sqrt(sqr_dist);
}

fn is_in_flower(pos: vec2<f32>) -> bool {
    let rand = hash12(pos * hash11(layer_height));
    return rand > 0.999;
}

@fragment
fn fragment(
    in: CuravtureMeshVertexOutput,
) -> @location(0) vec4<f32> {

    if layer_height == 0.0 {
        discard;
    }

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
    let x = n_vec4[0];
    let z = n_vec4[1];
    let normal = vec3<f32>(x, sqrt(1.0 - x*x - z*z), z);
    pbr.world_normal = normal;
    pbr.N = pbr.world_normal;
    pbr.V = pbr_functions::calculate_view(in.world_position, false);

    let height_frac = in.original_world_position.y / max_height + 0.0;
    if (height_frac >= grass_line) {
        discard;
    }

    let leaf = floor(in.world_position.xz * 50.0f);

    if is_in_path(in.original_world_position.xz) {
        discard;
    }

    if is_in_flower(leaf / 2.0) {
        pbr.material.base_color = vec4<f32>(149.0/255.0, 68.0/255.0, 166.0/255.0, 1.0);
    } else {
        let rand = hash12(leaf * layer_height);
        let norm_l_height = layer_height / 2.5f;
        let dist = min(3.0 * dist_to_path(leaf / 50.0f) / (PATH_WIDTH / 2.0), 1.0);
        if rand < 0.8 * dist {
            pbr.material.base_color = grass_color * norm_l_height * 1.1;
        } else {
            discard;
        }
    }

    var comp_col = hash12(vec2<f32>(floor(in.original_world_position.xz / COMPONENT_SIZE)));
    pbr.material.base_color *= comp_col;

    pbr.material.perceptual_roughness = 0.95;
    pbr.material.reflectance = 0.02;
    var output_color = pbr_functions::apply_pbr_lighting(pbr);
    output_color = pbr_functions::apply_fog(fog, output_color, in.world_position.xyz, view.world_position.xyz);

    return output_color;
}
