//! Maze game

// use std::collections::HashMap;
use bevy::core_pipeline::bloom::{BloomCompositeMode, BloomPrefilterSettings, BloomSettings};
use bevy::prelude::Color;
use bevy::window::WindowResolution;
use std::f32::consts::PI;
use std::net::Ipv4Addr;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
// use std::sync::{Arc, Mutex};

use crate::player_controller::{
    mouse_look, movement_input, pan_orbit_camera, toggle_cursor_lock, PanOrbitCamera, PlayerBody,
    PlayerCam,
};
use crate::shaders::{
    MazeLayerMaterial, MazeLayerMaterialDataHolder, TerrainMaterial, TerrainMaterialDataHolder,
    TerrainPlugin, MAX_TRIANGLES, MAX_VERTICES,
};
// use crate::terrain_loader::get_chunk;
use crate::terrain_render::{
    create_base_lattice, create_base_lattice_with_verts, create_lattice_plane,
    create_terrain_height_map, create_terrain_mesh, create_terrain_normal_map, MainTerrain, SCALE,
    TERRAIN_VERTICES, TEXTURE_SCALE, X_VIEW_DISTANCE, X_VIEW_DIST_M, Z_VIEW_DISTANCE,
    Z_VIEW_DIST_M,
};
use bevy::log::LogPlugin;
use bevy::math::{DVec2, Vec3};
// use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::pbr::{CascadeShadowConfigBuilder, ExtendedMaterial, NotShadowCaster};
use bevy::prelude::*;
use bevy::reflect::DynamicTypePath;
// use bevy::render::render_resource::{
//     Extent3d, TextureDimension, TextureFormat, VertexFormat, WgpuFeatures,
// };
// use bevy::render::settings::RenderCreation::Automatic;
// use bevy::render::settings::WgpuSettings;
use crate::maze_loader::MazeCellDataState::Invalid;
use crate::maze_loader::{
    setup_maze_loader, stream_maze_mesh, MazeDataHolder, MAZE_CELLS_X, MAZE_CELLS_Y,
    MAZE_DATA_COUNT,
};
use crate::maze_render::GetWall;
use crate::terrain_loader::{
    setup_terrain_loader, setup_transform_res, stream_terrain_mesh, update_transform_res,
    MainTerrainColldier, TerrainDataMap, TerrainTransformMaterialRes,
};
use crate::ui::*;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::pbr::wireframe::{WireframeColor, WireframeConfig, WireframePlugin};
use bevy::render::render_resource::{
    Buffer, BufferUsages, BufferVec, Extent3d, OwnedBindingResource, PreparedBindGroup,
    StorageBuffer, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::settings::RenderCreation::Automatic;
use bevy::render::settings::{WgpuFeatures, WgpuSettings};
use bevy::render::texture::{ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::render::RenderPlugin;
use bevy_atmosphere::prelude::*;
use bevy_framepace::{FramepacePlugin, FramepaceSettings, Limiter};
use bevy_mod_debugdump::render_graph::Settings;
use bevy_mod_wanderlust::{ControllerBundle, ControllerPhysicsBundle, WanderlustPlugin};
use bevy_rapier3d::prelude::*;
use bitvec::macros::internal::funty::Floating;
use delaunator::{triangulate, Point};
use futures_util::{FutureExt, Stream, StreamExt};
use rand::Rng;
use server::connection::*;
use server::terrain_gen::{TerrainGenerator, FOOTHILL_START, MAX_HEIGHT, TILE_SIZE};
use server::util::{cart_to_polar, lin_map, lin_map32};
use wasm_bindgen::prelude::wasm_bindgen;

mod fabian;
mod maze_render;
mod player_controller;
mod render;
mod shaders;
// mod terrain_loader;
mod maze_loader;
mod terrain_loader;
mod terrain_render;
mod test_render;
mod tests;
mod tree_render;
mod ui;

/// Spawn the player's collider and camera
fn spawn_player(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
) {
    // let material = mats.add(Color::WHITE.into());
    // commands
    //     .spawn((
    //         ControllerBundle {
    //             transform: Transform::from_xyz(0., 100., 0.),
    //             physics: ControllerPhysicsBundle { ..default() },
    //             ..default()
    //         },
    //         ColliderMassProperties::Density(50.0),
    //         Name::from("Player"),
    //         PlayerBody,
    //     ))
    //     .with_children(|commands| {
    //         commands
    //             .spawn((
    //                 Camera3dBundle {
    //                     transform: Transform::from_xyz(0.0, 0.5, 0.0),
    //                     projection: Projection::Perspective(PerspectiveProjection {
    //                         fov: 85.0 * (PI / 180.0),
    //                         aspect_ratio: 1.0,
    //                         near: 0.1 * 3.0,
    //                         far: 200.0 * 1000.0,
    //                     }),
    //                     camera: Camera {
    //                         // hdr: true,
    //                         ..default()
    //                     },
    //                     ..default()
    //                 },
    //                 AtmosphereCamera::default(),
    //                 FogSettings {
    //                     color: Color::rgba(0.1, 0.2, 0.4, 1.0),
    //                     directional_light_color: Color::rgba(1.0, 0.95, 0.75, 0.9),
    //                     directional_light_exponent: 60.0,
    //                     falloff: FogFalloff::from_visibility_colors(
    //                         100.0 * 1000.0, // distance in world units up to which objects retain visibility (>= 5% contrast)
    //                         Color::rgb(0., 0.8, 1.0), // atmospheric extinction color (after light is lost due to absorption by atmospheric particles)
    //                         Color::rgba(0., 0.8, 1.0, 1.0), // atmospheric inscattering color (light gained due to scattering from the sun)
    //                     ),
    //                 },
    //                 PlayerCam,
    //             ))
    //             .with_children(|commands| {
    //                 let mesh = meshes.add(shape::Cube { size: 0.5 }.into());
    //                 commands.spawn(PbrBundle {
    //                     mesh,
    //                     material: material.clone(),
    //                     transform: Transform::from_xyz(0.0, 0.0, -0.5),
    //                     ..default()
    //                 });
    //             });
    //     });
}

/// set up a simple 3D scene
fn add_lighting(mut commands: Commands /*mut wireframe_config: ResMut<WireframeConfig>*/) {
    // wireframe_config.global = true;
    let cascade_shadow_config = CascadeShadowConfigBuilder {
        first_cascade_far_bound: 0.3,
        // maximum_distance: 30000.,
        ..default()
    }
    .build();
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 100000.0,
            shadows_enabled: true,
            ..default()
        },
        cascade_shadow_config,
        transform: Transform::from_xyz(10., 100., 10.).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
    commands.insert_resource(ClearColor(Color::rgb(0., 0.8, 1.0)));
    commands.insert_resource(AtmosphereModel::new(Nishita {
        sun_intensity: 28.0,
        sun_position: Vec3::new(0.0, 1.0, 0.0),
        ..default()
    }));
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 2.0,
    });
}

fn maze_layer_height_func(layer: u32, layer_count: u32) -> f32 {
    let pos = lin_map32(0., layer_count as f32, 0.0, 1.0, layer as f32);
    fn F(p: f32) -> f32 {
        let bias = 0.0;
        ((2. * p - 1.) / 2.).atan() + p*bias + 0.5f32.atan()
    }
    let h = 2.0;
    h * F(pos) / F(1.)
}

fn create_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    // mut commands: Commands,
    // mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut maze_materials: ResMut<Assets<ExtendedMaterial<TerrainMaterial, MazeLayerMaterial>>>,
    mut textures: ResMut<Assets<Image>>,
    // player_controller: Query<Entity>,
    mut render_device: ResMut<RenderDevice>,
    mut render_queue: ResMut<RenderQueue>,
    mut terrain_material_data_holder: ResMut<TerrainMaterialDataHolder>,
    mut maze_data_holder: ResMut<MazeLayerMaterialDataHolder>,
) {
    let terrain_gen = TerrainGenerator::new();
    let lattice = create_base_lattice();
    commands.insert_resource(TerrainDataMap::new(&lattice));
    commands.insert_resource(MazeDataHolder::new());
    let terrain_mesh = create_terrain_mesh(Some(lattice));

    let mut heights: Vec<f32> = vec![];
    let dims = 100 / 2;
    for x in 0..dims {
        for z in 0..dims {
            let xp = lin_map(
                0.,
                dims as f64 - 1.,
                -TILE_SIZE / 4.,
                TILE_SIZE / 4.,
                x as f64,
            );
            let zp = lin_map(
                0.,
                dims as f64 - 1.,
                -TILE_SIZE / 4.,
                TILE_SIZE / 4.,
                z as f64,
            );
            heights.push(terrain_gen.get_height_for(xp, zp) as f32);
        }
    }

    commands.spawn((
        Collider::heightfield(
            heights,
            dims,
            dims,
            Vec3::new(TILE_SIZE as f32 / 2., 1., TILE_SIZE as f32 / 2.),
        ),
        MainTerrainColldier,
    ));

    let normal_handle = textures.add(create_terrain_normal_map(&terrain_gen));

    let mut delaunay_points: Vec<Point> = Vec::with_capacity(MAX_VERTICES);

    // let raw = &mut terrain_material_data_holder.vertices;//with_capacity(MAX_VERTICES * 2);
    // let heights = &mut terrain_material_data_holder.height;//with_capacity(MAX_VERTICES);
    // let gradients = &mut terrain_material_data_holder.gradients;//with_capacity(MAX_VERTICES * 2);

    // generate some random terrain data
    for v in create_lattice_plane(
        MAX_VERTICES as f64,
        X_VIEW_DIST_M * 1.1,
        Z_VIEW_DIST_M * 1.1,
    ) {
        // generate a random position in a circle
        let (r, theta) = cart_to_polar((v.x, v.z));
        let x = (r * SCALE).sinh() / SCALE * theta.cos();
        let z = (r * SCALE).sinh() / SCALE * theta.sin();
        let y = terrain_gen.get_height_for(x, z);
        delaunay_points.push(Point { x, y: z });
        terrain_material_data_holder.height.push(y as f32);
        terrain_material_data_holder.vertices.push(x as f32);
        terrain_material_data_holder.vertices.push(z as f32);
        let grad = terrain_gen.get_gradient(x, z);
        terrain_material_data_holder.gradients.push(grad.x as f32);
        terrain_material_data_holder.gradients.push(grad.y as f32);
    }

    let triangulation = triangulate(&delaunay_points);
    let vtx_count = terrain_mesh.count_vertices();

    let material = mats.add(Color::WHITE.into());

    // let mut triangles = &terrain_material_data_holder.triangles;
    // let mut halfedges = &terrain_material_data_holder.halfedges;

    triangulation.triangles.iter().for_each(|e| {
        terrain_material_data_holder.triangles.push(*e as u32);
        return;
    });
    triangulation.halfedges.iter().for_each(|e| {
        terrain_material_data_holder.halfedges.push(*e as u32);
        return;
    });

    terrain_material_data_holder.write_buffer(&*render_device, &*render_queue);

    // let mut triangle_indices = &terrain_material_data_holder.triangle_indices;
    terrain_material_data_holder
        .triangle_indices
        .values_mut()
        .append(&mut vec![0u32; vtx_count]);
    terrain_material_data_holder
        .triangle_indices
        .write_buffer(&*render_device, &*render_queue);

    terrain_material_data_holder
        .mesh_vertices
        .values_mut()
        .append(&mut vec![0f32; terrain_mesh.count_vertices()]);
    terrain_material_data_holder
        .mesh_vertices
        .write_buffer(&*render_device, &*render_queue);

    let base_material = TerrainMaterial {
        max_height: MAX_HEIGHT as f32 / 2.0,
        grass_line: 0.15,
        tree_line: 0.5,
        snow_line: 0.75,
        grass_color: Color::from([0.1, 0.4, 0.2, 1.]),
        tree_color: Color::from([0.2, 0.5, 0.25, 1.]),
        snow_color: Color::from([0.95, 0.95, 0.95, 1.]),
        stone_color: Color::from([0.34, 0.34, 0.34, 1.]),
        cosine_max_snow_slope: (45. * PI / 180.).cos(),
        cosine_max_tree_slope: (40. * PI / 180.).cos(),
        u_bound: ((X_VIEW_DIST_M * TEXTURE_SCALE).asinh() / TEXTURE_SCALE) as f32,
        v_bound: ((Z_VIEW_DIST_M * TEXTURE_SCALE).asinh() / TEXTURE_SCALE) as f32,
        normal_texture: normal_handle.into(),
        scale: TEXTURE_SCALE as f32,
        triangles: terrain_material_data_holder
            .triangles
            .buffer()
            .unwrap()
            .clone(),
        halfedges: terrain_material_data_holder
            .halfedges
            .buffer()
            .unwrap()
            .clone(),
        vertices: terrain_material_data_holder
            .vertices
            .buffer()
            .unwrap()
            .clone(),
        height: terrain_material_data_holder
            .height
            .buffer()
            .unwrap()
            .clone(),
        triangle_indices: terrain_material_data_holder
            .triangle_indices
            .buffer()
            .unwrap()
            .clone(),
        gradients: terrain_material_data_holder
            .gradients
            .buffer()
            .unwrap()
            .clone(),
        mesh_vertices: terrain_material_data_holder
            .mesh_vertices
            .buffer()
            .unwrap()
            .clone(),
        transform: Vec2::ZERO,
    };

    commands
        .spawn((
            ControllerBundle {
                transform: Transform::from_xyz(0., 300., 0.),
                physics: ControllerPhysicsBundle { ..default() },
                ..default()
            },
            ColliderMassProperties::Density(50.0),
            Name::from("Player"),
            PlayerBody,
        ))
        .with_children(|commands| {
            commands
                .spawn((
                    Camera3dBundle {
                        transform: Transform::from_xyz(0.0, 0.5, 0.0),
                        projection: Projection::Perspective(PerspectiveProjection {
                            fov: 60.0 * (PI / 180.0),
                            aspect_ratio: 1.0,
                            near: 0.1 * 3.0,
                            far: 400.0 * 1000.0,
                        }),
                        camera: Camera {
                            hdr: false,
                            ..default()
                        },
                        ..default()
                    },
                    AtmosphereCamera::default(),
                    FogSettings {
                        color: Color::rgba(0.1, 0.2, 0.4, 1.0),
                        directional_light_color: Color::rgba(1.0, 0.95, 0.75, 0.9),
                        directional_light_exponent: 60.0,
                        falloff: FogFalloff::from_visibility_colors(
                            200.0 * 1000.0, // distance in world units up to which objects retain visibility (>= 5% contrast)
                            Color::rgb(0., 0.8, 1.0), // atmospheric extinction color (after light is lost due to absorption by atmospheric particles)
                            Color::rgba(0., 0.8, 1.0, 1.0), // atmospheric inscattering color (light gained due to scattering from the sun)
                        ),
                    },
                    BloomSettings {
                        intensity: 0.5,
                        low_frequency_boost: 0.0,
                        low_frequency_boost_curvature: 0.0,
                        high_pass_frequency: 1.0 / 3.0,
                        prefilter_settings: BloomPrefilterSettings {
                            threshold: 0.0,
                            threshold_softness: 0.0,
                        },
                        composite_mode: BloomCompositeMode::EnergyConserving,
                    },
                    PlayerCam,
                ))
                .with_children(|commands| {
                    let mesh = meshes.add(shape::Cube { size: 0.5 }.into());
                    commands.spawn(PbrBundle {
                        mesh,
                        material: material.clone(),
                        transform: Transform::from_xyz(0.0, 0.0, -0.5),
                        ..default()
                    });
                });
        });

    commands.spawn((
        MaterialMeshBundle {
            mesh: meshes.add(terrain_mesh.clone()),
            transform: Transform::from_xyz(0.0, 0.0, 0.0),
            material: materials.add(base_material.clone()),
            ..default()
        },
        MainTerrain,
    ));

    maze_data_holder
        .raw_maze_data
        .write_buffer(&*render_device, &*render_queue);

    let buf = maze_data_holder.raw_maze_data.buffer().unwrap().clone();
    const LAYER_COUNT: u32 = 24;
    for layer in 0..LAYER_COUNT + 1 {
        let maze_layer_mesh = create_terrain_mesh(Some(create_base_lattice_with_verts(
            TERRAIN_VERTICES as f64,
            lin_map(
                0.,
                LAYER_COUNT as f64,
                10.,
                FOOTHILL_START + 1000.,
                layer as f64,
            ),
        )));
        
        let height = maze_layer_height_func(layer, LAYER_COUNT);
        println!("height {height}");
        commands.spawn((
            MaterialMeshBundle {
                mesh: meshes.add(maze_layer_mesh),
                transform: Transform::from_xyz(0.0, 0.0, 0.0),
                material: maze_materials.add(ExtendedMaterial {
                    base: base_material.clone(),
                    extension: MazeLayerMaterial {
                        layer_height: height,
                        data: maze_data_holder.raw_maze_data.buffer().unwrap().clone(),
                        maze_top_left: Vec2::ZERO,
                    },
                }),
                ..default()
            },
            MainTerrain,
        ));
    }

    // commands.entity(controller).push_children(&[terrain]);
}

pub fn spawn_maze(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
}

/// This system let's you toggle various wireframe settings
fn update_colors(
    keyboard_input: Res<Input<KeyCode>>,
    mut config: ResMut<WireframeConfig>,
    mut wireframe_colors: Query<&mut WireframeColor>,
    mut text: Query<&mut Text>,
) {
    text.single_mut().sections[0].value = format!(
        "
Controls
---------------
Z - Toggle global
X - Change global color
C - Change color of the green cube wireframe

WireframeConfig
-------------
Global: {}
Color: {:?}
",
        config.global, config.default_color,
    );

    // Toggle showing a wireframe on all meshes
    if keyboard_input.just_pressed(KeyCode::Z) {
        config.global = !config.global;
    }

    // Toggle the global wireframe color
    if keyboard_input.just_pressed(KeyCode::X) {
        config.default_color = if config.default_color == Color::WHITE.into() {
            Color::PINK.into()
        } else {
            Color::WHITE.into()
        };
    }

    // Toggle the color of a wireframe using WireframeColor and not the global color
    if keyboard_input.just_pressed(KeyCode::C) {
        for mut color in &mut wireframe_colors {
            color.color = if color.color == Color::LIME_GREEN.into() {
                Color::RED.into()
            } else {
                Color::LIME_GREEN.into()
            };
        }
    }
}

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub struct TerrainLoadingSystemSet;

fn main() {
    // init_panic_hook();

    let mut app = App::new();

    app
        // .insert_resource(ClearColor(Color::rgb(0., 0., 0.)))
        // the sky color
        // .insert_resource(ClearColor(Color::rgb(0.5294, 0.8078, 0.9216)))
        .add_plugins(bevy_tokio_tasks::TokioTasksPlugin::default())
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin {
                    default_sampler: ImageSamplerDescriptor {
                        // address_mode_u: AddressMode::Repeat,
                        // address_mode_v: AddressMode::Repeat,
                        // address_mode_w: AddressMode::Repeat,
                        mag_filter: ImageFilterMode::Linear,
                        ..Default::default()
                    },
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: WindowResolution::new(640., 480.)
                            .with_scale_factor_override(1.0),
                        ..default()
                    }),
                    ..default()
                })
                .disable::<LogPlugin>()
                .set(RenderPlugin {
                    render_creation: Automatic(WgpuSettings {
                        features: WgpuFeatures::POLYGON_MODE_LINE,
                        ..default()
                    }),
                }),
        )
        /*.add_plugins(
            ShaderUtilsPlugin,
        )*/
        .add_plugins(aether_spyglass::SpyglassPlugin)
        .add_plugins(FramepacePlugin)
        .add_plugins(AtmospherePlugin)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(WanderlustPlugin::default())
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .insert_resource(RapierConfiguration {
            timestep_mode: TimestepMode::Fixed {
                dt: 0.008,
                substeps: 4,
            },
            ..default()
        })
        .insert_resource(FramepaceSettings {
            limiter: Limiter::Manual(std::time::Duration::from_secs_f64(0.008)),
        })
        .insert_resource(TerrainMaterialDataHolder {
            triangles: BufferVec::new(BufferUsages::STORAGE),
            halfedges: BufferVec::new(BufferUsages::STORAGE),
            vertices: BufferVec::new(BufferUsages::STORAGE),
            height: BufferVec::new(BufferUsages::STORAGE),
            gradients: BufferVec::new(BufferUsages::STORAGE),
            triangle_indices: BufferVec::new(BufferUsages::STORAGE),
            mesh_vertices: BufferVec::new(BufferUsages::STORAGE),
        })
        .insert_resource(MazeLayerMaterialDataHolder {
            raw_maze_data: StorageBuffer::from(Box::new(
                [[[0u32; MAZE_DATA_COUNT]; MAZE_CELLS_Y]; MAZE_CELLS_X],
            )),
        })
        // .add_plugins(RapierDebugRenderPlugin::default())
        .add_plugins(WireframePlugin)
        .add_systems(Startup, add_lighting)
        .add_systems(Startup, spawn_player)
        .add_systems(Update, (movement_input, mouse_look, toggle_cursor_lock))
        // .add_systems(RunFixedUpdateLoop, pan_orbit_camera)
        // we do this last because the terrain rendering is attached to the player!
        .add_systems(PreStartup, create_terrain)
        // .add_systems(Startup, spawn_maze)
        .add_systems(PostStartup, setup_terrain_loader)
        .add_systems(PostStartup, setup_maze_loader)
        .add_systems(Startup, setup_transform_res)
        .add_systems(
            Update,
            (update_transform_res, stream_terrain_mesh, stream_maze_mesh),
        )
        .add_plugins(TerrainPlugin {});

    app.add_systems(Startup, setup_fps_counter);
    app.add_systems(Update, (fps_text_update_system, fps_counter_showhide));

    // app.add_systems(Update, update_colors);

    app.run();
}
