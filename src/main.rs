//! Maze game

use std::f32::consts::PI;

use crate::player_controller::{
    mouse_look, movement_input, pan_orbit_camera, toggle_cursor_lock, PanOrbitCamera, PlayerBody,
    PlayerCam,
};
use crate::shaders::{CurvaturePlugin, TerrainMaterial};
use crate::terrain_loader::get_chunk;
use crate::terrain_render::{
    create_base_terrain_mesh, create_terrain_mesh, create_terrain_normal_map, load_terrain_heights,
    X_VIEW_DISTANCE, X_VIEW_DIST_M, Z_VIEW_DISTANCE, Z_VIEW_DIST_M,
};
use bevy::app::RunFixedUpdateLoop;
use bevy::math::Vec3;
use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::pbr::{CascadeShadowConfigBuilder, NotShadowCaster};
use bevy::prelude::*;
use bevy::render::mesh::VertexAttributeValues::{Float32x2, Float32x3};
use bevy::render::render_resource::{FilterMode, SamplerDescriptor, WgpuFeatures};
use bevy::render::settings::WgpuSettings;
use bevy::render::RenderPlugin;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use bevy_atmosphere::prelude::*;
use bevy_framepace::{FramepacePlugin, FramepaceSettings, Limiter};
use bevy_mod_wanderlust::{
    ControllerBundle, ControllerInput, ControllerPhysicsBundle, WanderlustPlugin,
};
use bevy_rapier3d::prelude::*;
use bevy_shader_utils::ShaderUtilsPlugin;
use futures_util::{pin_mut, FutureExt, Stream, StreamExt};
use server::terrain_data::TerrainTile;
use server::terrain_gen::{TerrainGenerator, MAX_HEIGHT, TILE_SIZE};
use server::util::lin_map;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::{connect_async, connect_async_with_config};
use url::Url;

mod maze_gen;
mod maze_render;
mod player_controller;
mod render;
mod shaders;
mod terrain_loader;
mod terrain_render;
mod test_render;
mod tests;
mod tree_render;

/// Spawn the player's collider and camera
fn spawn_player(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
) {
    // let translation = Vec3::new(0.0, 2.5, -5.0);
    // let radius = translation.length();
    // commands.spawn((
    //     Camera3dBundle {
    //         transform: Transform::from_translation(translation).looking_at(Vec3::ZERO, Vec3::Y),
    //         ..Default::default()
    //     },
    //     PanOrbitCamera {
    //         radius,
    //         ..Default::default()
    //     },
    // ));

    let mesh = meshes.add(
        shape::Capsule {
            radius: 0.5,
            depth: 1.0,
            ..default()
        }
        .into(),
    );

    let material = mats.add(Color::WHITE.into());

    commands
        .spawn((
            ControllerBundle {
                physics: ControllerPhysicsBundle {
                    // Lock the axes to prevent camera shake whilst moving up slopes
                    locked_axes: LockedAxes::ROTATION_LOCKED,
                    restitution: Restitution {
                        coefficient: 0.0,
                        combine_rule: CoefficientCombineRule::Min,
                    },
                    collider: Collider::capsule(
                        Vec3::new(0.0, 0.0, 0.0),
                        Vec3::new(0.0, 2.0, 0.0),
                        0.5,
                    ),
                    ..default()
                },
                ..default()
            },
            ColliderMassProperties::Density(50.0),
            Name::from("Player"),
            PlayerBody,
        ))
        // .insert(PbrBundle {
        //     mesh,
        //     material: material.clone(),
        //     ..default()
        // })
        .with_children(|commands| {
            commands
                .spawn((
                    Camera3dBundle {
                        transform: Transform::from_xyz(0.0, 0.5, 0.0),
                        projection: Projection::Perspective(PerspectiveProjection {
                            fov: 85.0 * (PI / 180.0),
                            aspect_ratio: 1.0,
                            near: 0.1 * 3.0,
                            far: 200.0 * 1000.0,
                        }),
                        camera: Camera {
                            // hdr: true,
                            ..default()
                        },
                        ..default()
                    },
                    AtmosphereCamera::default(),
                    // FogSettings {
                    //     color: Color::rgba(0.1, 0.2, 0.4, 1.0),
                    //     directional_light_color: Color::rgba(1.0, 0.95, 0.75, 0.9),
                    //     directional_light_exponent: 60.0,
                    //     falloff: FogFalloff::from_visibility_colors(
                    //         100.0 * 1000.0, // distance in world units up to which objects retain visibility (>= 5% contrast)
                    //         Color::rgb(0., 0.8, 1.0), // atmospheric extinction color (after light is lost due to absorption by atmospheric particles)
                    //         Color::rgba(0., 0.8, 1.0, 1.0), // atmospheric inscattering color (light gained due to scattering from the sun)
                    //     ),
                    // },
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
}

/// set up a simple 3D scene
fn add_lighting(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    // mut wireframe_config: ResMut<WireframeConfig>,
) {
    // wireframe_config.global = true;
    let cascade_shadow_config = CascadeShadowConfigBuilder {
        first_cascade_far_bound: 0.3,
        maximum_distance: 3.0,
        ..default()
    }
    .build();
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 90000.0,
            shadows_enabled: true,
            ..default()
        },
        cascade_shadow_config,
        transform: Transform::from_xyz(100., 100., 100.).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
    commands.insert_resource(ClearColor(Color::rgb(0., 0.8, 1.0)));
    commands.insert_resource(AtmosphereModel::new(Nishita {
        sun_intensity: 25.0,
        ..default()
    }));
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 1.1,
    });
}

fn create_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut textures: ResMut<Assets<Image>>,
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
    // let terrain_mesh = create_terrain_mesh(&terrain_gen);

    let mut heights: Vec<Real> = vec![];
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

    commands.spawn(Collider::heightfield(
        heights,
        dims,
        dims,
        Vec3::new(TILE_SIZE as f32 / 2., 1., TILE_SIZE as f32 / 2.),
    ));

    // let normal_handle = textures.add(create_terrain_normal_map(&terrain_gen));
    //
    // let x_bound = X_VIEW_DISTANCE as f64 * TILE_SIZE;
    // let z_bound = Z_VIEW_DISTANCE as f64 * TILE_SIZE;
    // commands.spawn(MaterialMeshBundle {
    //     mesh: meshes.add(terrain_mesh),
    //     material: materials.add(TerrainMaterial {
    //         max_height: MAX_HEIGHT as f32,
    //         grass_line: 0.15,
    //         grass_color: Color::from([0.1, 0.5, 0.2, 1.]),
    //         tree_line: 0.5,
    //         tree_color: Color::from([0.2, 0.6, 0.25, 1.]),
    //         snow_line: 0.75,
    //         snow_color: Color::from([0.95, 0.95, 0.95, 1.]),
    //         stone_color: Color::from([0.34, 0.34, 0.34, 1.]),
    //         cosine_max_snow_slope: (45. * PI / 180.).cos(),
    //         cosine_max_tree_slope: (40. * PI / 180.).cos(),
    //         u_bound: x_bound as f32,
    //         v_bound: z_bound as f32,
    //         normal_texture: normal_handle.into(),
    //     }),
    //     ..default()
    // });
}

#[derive(Component)]
struct LoadTerrain(Task<()>);

fn spawn_tasks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut textures: ResMut<Assets<Image>>,
) {
    let thread_pool = AsyncComputeTaskPool::get();
    // Spawn new task on the AsyncComputeTaskPool; the task will be
    // executed in the background, and the Task future returned by
    // spawn() can be used to poll for the result
    let task = thread_pool.spawn(async move {});

    // Spawn new entity and add our new task as a component
    commands.spawn(LoadTerrain(task));
}

fn load_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut textures: ResMut<Assets<Image>>,
) {
    println!("Load terrain start");
    // Rust tells me this won't run unless I await it or whatever but I can't since load_terrain is async
    let mesh = create_base_terrain_mesh();
    let mut mesh_handle = meshes.add(mesh);
    // TODO: put this on the server!
    let normal_handle = textures.add(create_terrain_normal_map(&TerrainGenerator::new()));
    commands.spawn(MaterialMeshBundle {
        mesh: mesh_handle.clone(),
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
            u_bound: X_VIEW_DIST_M as f32,
            v_bound: Z_VIEW_DIST_M as f32,
            normal_texture: normal_handle.clone().into(),
        }),
        ..default()
    });

    async {
        let res = connect_async(
            Url::parse("ws://127.0.0.1:9002").expect("Can't connect to case count URL"),
        )
        .await;
        println!("Try connect 2");
        if let Ok((mut ws_stream, response)) = res {
            let mesh = meshes.get_mut(&mut mesh_handle).unwrap();
            let pos;
            if let Some(Float32x3(p)) = mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION) {
                pos = p;
            } else {
                panic!("Unexpected position attribute");
            }

            let terrain_mesh_stream = load_terrain_heights(pos.clone(), &mut ws_stream);
            pin_mut!(terrain_mesh_stream);

            while let Some(vtx) = terrain_mesh_stream.next().await {
                // according to scientists this should modify the mesh in real time
                // TODO: verify
                pos[vtx.0][1] = vtx.1[1];
            }
        } else {
            println!("Fail");
        }
    };

    return;
}

fn main() {
    let _ = App::new()
        // .insert_resource(ClearColor(Color::rgb(0., 0., 0.)))
        // the sky color
        // .insert_resource(ClearColor(Color::rgb(0.5294, 0.8078, 0.9216)))
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
            aether_spyglass::SpyglassPlugin,
            FramepacePlugin,
        ))
        .add_plugins(AtmospherePlugin)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(WanderlustPlugin)
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
        .add_systems(Startup, create_terrain)
        .add_systems(Startup, add_lighting)
        .add_systems(Startup, spawn_player)
        .add_systems(Startup, load_terrain)
        // .add_systems(RunFixedUpdateLoop, pan_orbit_camera)
        .add_systems(Update, (movement_input, mouse_look, toggle_cursor_lock))
        .run();
}
