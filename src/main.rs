//! Maze game

// use std::collections::HashMap;
use std::f32::consts::PI;
use std::net::Ipv4Addr;
use std::time::Duration;
use bevy::core_pipeline::bloom::{BloomCompositeMode, BloomSettings};
// use std::sync::{Arc, Mutex};

use crate::player_controller::{
    mouse_look, movement_input, pan_orbit_camera, toggle_cursor_lock, PanOrbitCamera, PlayerBody,
    PlayerCam,
};
use crate::shaders::{
    ParticleSystem, TerrainMaterial, TerrainPlugin, HEIGHT, MAX_TRIANGLES, MAX_VERTICES, WIDTH,
};
// use crate::terrain_loader::get_chunk;
use crate::terrain_render::{
    create_base_lattice, create_base_terrain_mesh, create_lattice_plane, create_terrain_height_map,
    create_terrain_mesh, create_terrain_normal_map, MainTerrain, SCALE, TERRAIN_VERTICES,
    TEXTURE_SCALE, X_VIEW_DISTANCE, X_VIEW_DIST_M, Z_VIEW_DISTANCE, Z_VIEW_DIST_M,
};
use bevy::log::LogPlugin;
use bevy::math::{DVec2, Vec3};
// use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::pbr::{CascadeShadowConfigBuilder, NotShadowCaster};
use bevy::prelude::*;
use bevy::reflect::DynamicTypePath;
// use bevy::render::render_resource::{
//     Extent3d, TextureDimension, TextureFormat, VertexFormat, WgpuFeatures,
// };
// use bevy::render::settings::RenderCreation::Automatic;
// use bevy::render::settings::WgpuSettings;
use crate::maze_gen::{
    populate_maze, SquareMaze, SquareMazeComponent, SQUARE_MAZE_CELL_SIZE, SQUARE_MAZE_WALL_WIDTH,
};
use crate::maze_render::GetWall;
use crate::terrain_loader::{
    setup_terrain_loader, setup_transform_res, stream_terrain_mesh, update_transform_res,
    MainTerrainColldier, TerrainDataMap,
};
use crate::ui::*;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::texture::{ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
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
use server::terrain_gen::{TerrainGenerator, MAX_HEIGHT, TILE_SIZE};
use server::util::{cart_to_polar, lin_map};
use wasm_bindgen::prelude::wasm_bindgen;

mod fabian;
mod maze_gen;
mod maze_render;
mod player_controller;
mod render;
mod shaders;
// mod terrain_loader;
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

fn create_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    // mut commands: Commands,
    // mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut textures: ResMut<Assets<Image>>,
    // player_controller: Query<Entity>,
) {
    let terrain_gen = TerrainGenerator::new();
    let lattice = create_base_lattice();
    commands.insert_resource(TerrainDataMap::new(&lattice));
    let terrain_mesh = create_base_terrain_mesh(Some(lattice));

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
    let mut raw: Vec<f32> = Vec::with_capacity(MAX_VERTICES * 2);
    let mut heights: Vec<f32> = Vec::with_capacity(MAX_VERTICES);
    let mut gradients: Vec<f32> = Vec::with_capacity(MAX_VERTICES * 2);

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
        heights.push(y as f32);
        raw.push(x as f32);
        raw.push(z as f32);
        let grad = terrain_gen.get_gradient(x, z);
        gradients.push(grad.x as f32);
        gradients.push(grad.y as f32);
    }

    let triangulation = triangulate(&delaunay_points);
    let vtx_count = terrain_mesh.count_vertices();

    let material = mats.add(Color::WHITE.into());
    commands
        .spawn((
            ControllerBundle {
                transform: Transform::from_xyz(0., 2000., 0.),
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
                            hdr: true,
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
                        ..default()
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
            commands.spawn((
                MaterialMeshBundle {
                    mesh: meshes.add(terrain_mesh),
                    transform: Transform::from_xyz(0.0, 0.0, 0.0),
                    material: materials.add(TerrainMaterial {
                        max_height: MAX_HEIGHT as f32,
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
                        triangles: triangulation.triangles.iter().map(|e| *e as u32).collect(),
                        halfedges: triangulation.halfedges.iter().map(|e| *e as u32).collect(),
                        vertices: raw,
                        height: heights,
                        triangle_indices: vec![0u32; vtx_count],
                        gradients,
                    }),
                    ..default()
                },
                MainTerrain,
            ));
        });

    // commands.entity(controller).push_children(&[terrain]);
}

pub fn spawn_maze(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    let mut graph = SquareMaze {
        maze: SquareMazeComponent::new(),
        cell_size: SQUARE_MAZE_CELL_SIZE,
        offset: (0, 0),
        size: 32,
        wall_width: SQUARE_MAZE_WALL_WIDTH,
    };
    let mut starting_comp = SquareMazeComponent::new();
    starting_comp.add_node((-1, 0));
    populate_maze(&mut graph, vec![starting_comp]);

    graph.save();
    let graph = SquareMaze::load(graph.offset);

    // leaf texture
    // let texture_handle = load_tiled_texture(&mut images, "hedge.png");
    let texture_handle = asset_server.load("hedge.png");

    for mesh in graph.get_wall_geometry(1.0, 2.0) {
        commands.spawn(PbrBundle {
            mesh: meshes.add(mesh),
            material: materials.add(StandardMaterial {
                // base_color: Color::rgba(0.01, 0.8, 0.2, 1.0).into(),
                base_color_texture: Some(texture_handle.clone()),
                alpha_mode: AlphaMode::Mask(0.5),
                ..default()
            }),
            transform: Transform::from_xyz(0.0, 2.0, 0.0),
            ..default()
        });
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
                .disable::<LogPlugin>(), // .set(RenderPlugin {
                                         //     render_creation: Automatic(WgpuSettings {
                                         //         features: WgpuFeatures::POLYGON_MODE_LINE,
                                         //         ..default()
                                         //     }),
                                         // }),
        )
        .add_plugins(MaterialPlugin::<TerrainMaterial> { ..default() })
        /*.add_plugins(
            ShaderUtilsPlugin,
        )*/
        .add_plugins(TerrainPlugin {})
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
        // .add_plugins(RapierDebugRenderPlugin::default())
        // .add_plugins(WireframePlugin)
        .add_systems(Startup, add_lighting)
        .add_systems(Startup, spawn_player)
        .add_systems(Update, (movement_input, mouse_look, toggle_cursor_lock))
        // .add_systems(RunFixedUpdateLoop, pan_orbit_camera)
        // we do this last because the terrain rendering is attached to the player!
        .add_systems(PreStartup, create_terrain)
        .add_systems(Startup, spawn_maze)
        .add_systems(PostStartup, setup_terrain_loader)
        .add_systems(Startup, setup_transform_res)
        .add_systems(Update, (update_transform_res, stream_terrain_mesh));

    app.add_systems(Startup, setup_fps_counter);
    app.add_systems(Update, (fps_text_update_system, fps_counter_showhide));

    /*
    Begin logic_compute_shader main.rs
     */
    app.add_systems(Startup, setup)
        .add_systems(Update, spawn_on_space_bar);
    /*
    End logic_compute_shader main.rs
     */

    app.run();
}

/*
Begin logic_compute_shader main.rs
 */
fn create_texture(images: &mut Assets<Image>) -> Handle<Image> {
    let mut image = Image::new_fill(
        Extent3d {
            width: WIDTH as u32,
            height: HEIGHT as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8Unorm,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    image.sampler = ImageSampler::nearest();
    images.add(image)
}

fn spawn_on_space_bar(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    keyboard: Res<Input<KeyCode>>,
) {
    if keyboard.pressed(KeyCode::Space) {
        let image = create_texture(&mut images);
        commands
            .spawn(SpriteBundle {
                sprite: Sprite {
                    custom_size: Some(Vec2::new(WIDTH * 3.0, HEIGHT * 3.0)),
                    ..default()
                },
                texture: image.clone(),
                ..default()
            })
            .insert(ParticleSystem {
                rendered_texture: image,
            });
    }
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = create_texture(&mut images);
    commands
        .spawn(SpriteBundle {
            sprite: Sprite {
                custom_size: Some(Vec2::new(WIDTH * 3.0, HEIGHT * 3.0)),
                ..default()
            },
            texture: image.clone(),
            ..default()
        })
        .insert(ParticleSystem {
            rendered_texture: image,
        });

    // commands.spawn(Camera2dBundle::default());
}
/*
end main.rs
 */
