//! A simple 3D scene with light shining over a cube sitting on a plane.

mod maze_gen;
mod maze_render;
mod render;
mod terrain_render;
mod test_render;
mod tree_render;

use crate::maze_gen::{
    populate_maze, CircleMaze, CircleMazeComponent, CircleNode, Maze, SquareMaze,
    SquareMazeComponent, SquareNode, SQUARE_MAZE_CELL_SIZE, SQUARE_MAZE_WALL_WIDTH,
};
use crate::maze_render::{
    get_arc_mesh, get_segment_mesh, polar_to_cart, Arc, Circle, GetWall, Segment,
};
use crate::render::SimpleVertex;
use crate::terrain_render::{
    create_lattice_plane, get_scaled_normal_map, get_terrain_mesh, get_tile_dist,
    transform_lattice_positions, update_terrain_mesh, TILE_WORLD_SIZE, VIEW_DISTANCE,
};
use crate::test_render::{
    draw_circle, draw_segment, to_canvas_space, AxisTransform, DrawableCircle, DrawableSegment,
};
use crate::tree_render::get_tree_mesh;
use bevy::app::RunFixedUpdateLoop;
use bevy::asset::load_internal_asset;
use bevy::input::keyboard::KeyboardInput;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::math::Vec3;
use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::{TypePath, TypeUuid};
use bevy::render::camera::Projection;
use bevy::render::mesh::{
    Indices, MeshVertexBufferLayout, PrimitiveTopology, VertexAttributeValues,
};
use bevy::render::render_asset::RenderAsset;
use bevy::render::render_resource::{
    AddressMode, AsBindGroup, Extent3d, FilterMode, RenderPipelineDescriptor, SamplerDescriptor,
    ShaderRef, SpecializedMeshPipelineError,
};
use bevy::render::settings::{WgpuFeatures, WgpuSettings};
use bevy::render::texture::ImageSampler::Descriptor;
use bevy::render::texture::{CompressedImageFormats, ImageSampler, ImageType};
use bevy::render::RenderPlugin;
use bevy::window::PrimaryWindow;
use bevy_shader_utils::ShaderUtilsPlugin;
use image::io::Reader as ImageReader;
use image::{DynamicImage, ImageBuffer, Rgb, Rgb32FImage, RgbImage};
use imageproc::drawing::{draw_text, draw_text_mut};
use itertools::iproduct;
use rand::{thread_rng, Rng, RngCore, SeedableRng};
use rusttype::{Font, Scale};
use server::terrain_gen::{
    generate_tile, TerrainGenerator, MAX_HEIGHT, TILE_RESOLUTION, TILE_SIZE, VIEW_RADIUS,
};
use server::util::lin_map32;
use std::f32::consts::PI;
use std::fmt::format;
use std::io::Cursor;
use bevy_rapier3d::geometry::{Collider, Restitution};
use bevy_rapier3d::plugin::RapierPhysicsPlugin;
use bevy_rapier3d::prelude::{NoUserData, RapierDebugRenderPlugin, RigidBody};

#[derive(AsBindGroup, Debug, Clone, TypeUuid, TypePath)]
#[uuid = "b62bb455-a72c-4b56-87bb-81e0554e234f"]
pub struct TerrainMaterial {
    #[uniform(0)]
    max_height: f32,
    #[uniform(1)]
    grass_line: f32,
    #[uniform(2)]
    tree_line: f32,
    #[uniform(3)]
    snow_line: f32,
    #[uniform(4)]
    grass_color: Color,
    #[uniform(5)]
    tree_color: Color,
    #[uniform(6)]
    snow_color: Color,
    #[uniform(7)]
    stone_color: Color,
    #[uniform(8)]
    cosine_max_snow_slope: f32,
    #[uniform(9)]
    cosine_max_tree_slope: f32,
    // #[texture(10)]
    // #[sampler(11)]
    // normal_texture: Option<Handle<Image>>,
}

/// The Material trait is very configurable, but comes with sensible defaults for all methods.
/// You only need to implement functions for features that need non-default behavior. See the Material api docs for details!
impl Material for TerrainMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/curvature_transform.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/terrain_color.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayout,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

// /// Tags an entity as capable of panning and orbiting.
#[derive(Component)]
struct PanOrbitCamera {
    /// The "focus point" to orbit around. It is automatically updated when panning the camera
    pub focus: Vec3,
    pub radius: f32,
    pub upside_down: bool,
}

impl Default for PanOrbitCamera {
    fn default() -> Self {
        PanOrbitCamera {
            focus: Vec3::ZERO,
            radius: 5.0,
            upside_down: false,
        }
    }
}

/// Pan the camera with middle mouse click, zoom with scroll wheel, orbit with right mouse click.
fn pan_orbit_camera(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut ev_motion: EventReader<MouseMotion>,
    mut ev_scroll: EventReader<MouseWheel>,
    input_mouse: Res<Input<MouseButton>>,
    input_keyboard: Res<Input<KeyCode>>,
    mut query: Query<(&mut PanOrbitCamera, &mut Transform, &Projection)>,
) {
    StandardMaterial { ..default() };
    // change input mapping for orbit and panning here
    let orbit_button = MouseButton::Right;
    let pan_button = MouseButton::Middle;

    let mut pan = Vec2::ZERO;
    let mut rotation_move = Vec2::ZERO;
    let mut scroll = 0.0;
    let mut orbit_button_changed = false;

    if input_mouse.pressed(orbit_button) {
        for ev in ev_motion.iter() {
            rotation_move += ev.delta;
        }
    }
    if input_mouse.pressed(pan_button) || input_keyboard.just_pressed(KeyCode::Space) {
        // Pan only if we're not rotating at the moment
        for ev in ev_motion.iter() {
            pan += ev.delta;
        }
    }
    for ev in ev_scroll.iter() {
        scroll += ev.y;
    }
    if input_mouse.just_released(orbit_button) || input_mouse.just_pressed(orbit_button) {
        orbit_button_changed = true;
    }

    for (mut pan_orbit, mut transform, projection) in query.iter_mut() {
        if orbit_button_changed {
            // only check for upside down when orbiting started or ended this frame
            // if the camera is "upside" down, panning horizontally would be inverted, so invert the input to make it correct
            let up = transform.rotation * Vec3::Y;
            pan_orbit.upside_down = up.y <= 0.0;
        }

        let mut any = false;
        if rotation_move.length_squared() > 0.0 {
            any = true;
            let window = get_primary_window_size(&windows);
            let delta_x = {
                let delta = rotation_move.x / window.x * std::f32::consts::PI * 2.0;
                if pan_orbit.upside_down {
                    -delta
                } else {
                    delta
                }
            };
            let delta_y = rotation_move.y / window.y * std::f32::consts::PI;
            let yaw = Quat::from_rotation_y(-delta_x);
            let pitch = Quat::from_rotation_x(-delta_y);
            transform.rotation = yaw * transform.rotation; // rotate around global y axis
            transform.rotation = transform.rotation * pitch; // rotate around local x axis
        } else if pan.length_squared() > 0.0 {
            any = true;
            // make panning distance independent of resolution and FOV,
            let window = get_primary_window_size(&windows);
            if let Projection::Perspective(projection) = projection {
                pan *= Vec2::new(projection.fov * projection.aspect_ratio, projection.fov) / window;
            }
            // translate by local axes
            let right = transform.rotation * Vec3::X * -pan.x;
            let up = transform.rotation * Vec3::Y * pan.y;
            // make panning proportional to distance away from focus point
            let translation = (right + up) * pan_orbit.radius;
            pan_orbit.focus += translation;
        } else if scroll.abs() > 0.0 {
            any = true;
            pan_orbit.radius -= 10. * scroll;
            // dont allow zoom to reach zero or you get stuck
            //pan_orbit.radius = f32::max(pan_orbit.radius, 0.05);
        }

        if any {
            // emulating parent/child to make the yaw/y-axis rotation behave like a turntable
            // parent = x and y rotation
            // child = z-offset
            let rot_matrix = Mat3::from_quat(transform.rotation);
            transform.translation =
                pan_orbit.focus + rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, pan_orbit.radius));
        }
    }

    // consume any remaining events, so they don't pile up if we don't need them
    // (and also to avoid Bevy warning us about not checking events every frame update)
    ev_motion.clear();
}

fn get_primary_window_size(windows: &Query<&Window, With<PrimaryWindow>>) -> Vec2 {
    let window = windows.get_single().unwrap();
    let window = Vec2::new(window.width() as f32, window.height() as f32);
    window
}

/// Spawn a camera like this
fn spawn_camera(mut commands: Commands) {
    let translation = Vec3::new(0.0, 30.0, 0.0);
    let radius = translation.length();

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(translation).looking_at(Vec3::ZERO, Vec3::Y),
            ..Default::default()
        },
        PanOrbitCamera {
            radius,
            ..Default::default()
        },
    ));
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    // mut wireframe_config: ResMut<WireframeConfig>,
) {
    // wireframe_config.global = true;
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 64000.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_xyz(75.0, 100.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 1.0,
    });

    // camera
    spawn_camera(commands)
}

fn create_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    // commands.spawn(PbrBundle {
    //     mesh: meshes.add(get_tree_mesh(0)),
    //     material: materials.add(StandardMaterial {
    //         base_color: Color::DARK_GREEN,
    //         double_sided: true,
    //         cull_mode: None,
    //         ..default()
    //     }),
    //     ..default()
    // });
    let terrain_gen = TerrainGenerator::new();
    let (mut verts, indices, colors) = create_lattice_plane();
    transform_lattice_positions(&mut verts);
    let tile_mesh = get_terrain_mesh(verts, indices, colors, &terrain_gen);
    commands.spawn(MaterialMeshBundle {
        mesh: meshes.add(tile_mesh),
        material: materials.add(TerrainMaterial {
            max_height: MAX_HEIGHT as f32,
            grass_line: 0.15,
            grass_color: Color::from([0.1, 0.5, 0.2, 1.]),
            tree_line: 0.5,
            tree_color: Color::from([0.2, 0.6, 0.25, 1.]),
            snow_line: 0.75,
            snow_color: Color::from([0.95, 0.95, 0.95, 1.]),
            stone_color: Color::from([0.34, 0.34, 0.34, 1.]),
            cosine_max_snow_slope: (45. * PI / 180.).cos(),
            cosine_max_tree_slope: (40. * PI / 180.).cos(),
            // normal_texture: normal_handle.into(),
        }),
        ..default()
    });
    // println!("Done all tiles, triangles is {}", triangles)
}

fn setup_physics(mut commands: Commands) {
    /* Create the ground. */
    commands
        .spawn(Collider::cuboid(100.0, 0.1, 100.0))
        .insert(TransformBundle::from(Transform::from_xyz(0.0, -2.0, 0.0)));

    /* Create the bouncing ball. */
    commands
        .spawn(RigidBody::Dynamic)
        .insert(Collider::ball(0.5))
        .insert(Restitution::coefficient(0.7))
        .insert(TransformBundle::from(Transform::from_xyz(0.0, 4.0, 0.0)));
}

pub const CURVATURE_MESH_VERTEX_OUTPUT: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 128741983741982);

struct CurvaturePlugin {}

impl Plugin for CurvaturePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            CURVATURE_MESH_VERTEX_OUTPUT,
            "../assets/shaders/curvature_mesh_vertex_output.wgsl",
            Shader::from_wgsl
        );
    }
}

fn main() {
    let _ = App::new()
        // .insert_resource(ClearColor(Color::rgb(0., 0., 0.)))
        .insert_resource(ClearColor(Color::rgb(0.5294, 0.8078, 0.9216)))
        .add_plugins((
            DefaultPlugins.set(ImagePlugin {
                default_sampler: SamplerDescriptor {
                    // address_mode_u: AddressMode::Repeat,
                    // address_mode_v: AddressMode::Repeat,
                    // address_mode_w: AddressMode::Repeat,
                    mag_filter: FilterMode::Linear,
                    ..Default::default()
                },
            }),
            // .set(RenderPlugin {
            //     wgpu_settings: WgpuSettings {
            //         features: WgpuFeatures::POLYGON_MODE_LINE,
            //         ..default()
            //     },
            // }),
            MaterialPlugin::<TerrainMaterial> { ..default() },
            ShaderUtilsPlugin,
            CurvaturePlugin {},
        ))
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default())
        // .add_plugins(WireframePlugin)
        .add_systems(Startup, setup)
        .add_systems(Startup, create_terrain)
        .add_systems(RunFixedUpdateLoop, pan_orbit_camera)
        .run();

    // let terrain_gen = TerrainGenerator::new();
    // let tile = generate_tile(&terrain_gen, 0, 0);
    // let mut img: RgbImage = ImageBuffer::new(tile.0.len() as u32, tile.0.len() as u32);
    //
    // for x in 0..img.width() {
    //     for y in 0..img.height() {
    //         let val = tile.0[x as usize][y as usize];
    //         let res = lin_map32(-2., 25., 0., 255., val).round() as u8;
    //         img.put_pixel(x, y, Rgb([res, res, res]));
    //     }
    // }
    //
    // img.save("terrain_out.png").unwrap();
}
