//! Maze game

// use std::collections::HashMap;
use std::f32::consts::PI;
// use std::sync::{Arc, Mutex};

use crate::player_controller::{
    mouse_look, movement_input, pan_orbit_camera, toggle_cursor_lock, PanOrbitCamera, PlayerBody,
    PlayerCam,
};
use crate::shaders::{TerrainMaterial, TerrainPlugin, MAX_TRIANGLES, MAX_VERTICES};
// use crate::terrain_loader::get_chunk;
use crate::terrain_render::{
    create_base_terrain_mesh, create_lattice_plane, create_terrain_height_map, create_terrain_mesh,
    create_terrain_normal_map, MainTerrain, Square, TerrainDataMap, SCALE, TERRAIN_VERTICES,
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
use bevy::render::texture::{ImageFilterMode, ImageSamplerDescriptor};
// use bevy::render::RenderPlugin;
// use bevy::tasks::{AsyncComputeTaskPool, IoTaskPool, Task};
use bevy_atmosphere::prelude::*;
use bevy_framepace::{FramepacePlugin, FramepaceSettings, Limiter};
// use bevy_mod_debugdump::print_render_graph;
use bevy_mod_wanderlust::{
    ControllerBundle, ControllerInput, ControllerPhysicsBundle, WanderlustPlugin,
};
use bevy_rapier3d::prelude::*;
// use bevy_rapier3d::rapier::crossbeam::channel::unbounded;
use bitvec::macros::internal::funty::Floating;
use delaunator::{triangulate, Point};
//use bevy_shader_utils::ShaderUtilsPlugin;
use futures_util::{pin_mut, FutureExt, Stream, StreamExt};
use rand::{thread_rng, Rng};
// use tokio::net::unix::SocketAddr;
// use tokio::net::{TcpListener, TcpStream};
// use tokio::sync::mpsc::Sender;
// use server::terrain_data::TerrainTile;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use server::terrain_gen::{TerrainGenerator, MAX_HEIGHT, TILE_SIZE};
use server::util::{cart_to_polar, lin_map};
// use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
// use tokio_tungstenite::{connect_async, connect_async_with_config};
// use url::Url;

use bevy::diagnostic::DiagnosticsStore;
use bevy::render::mesh::VertexAttributeValues::Float32x3;
use wasm_bindgen::prelude::wasm_bindgen;

/// Marker to find the container entity so we can show/hide the FPS counter
#[derive(Component)]
struct FpsRoot;

/// Marker to find the text entity so we can update it
#[derive(Component)]
struct FpsText;

fn setup_fps_counter(mut commands: Commands) {
    // create our UI root node
    // this is the wrapper/container for the text
    let root = commands
        .spawn((
            FpsRoot,
            NodeBundle {
                // give it a dark background for readability
                background_color: BackgroundColor(Color::BLACK.with_a(0.5)),
                // make it "always on top" by setting the Z index to maximum
                // we want it to be displayed over all other UI
                z_index: ZIndex::Global(i32::MAX),
                style: Style {
                    position_type: PositionType::Absolute,
                    // position it at the top-right corner
                    // 1% away from the top window edge
                    right: Val::Percent(1.),
                    top: Val::Percent(1.),
                    // set bottom/left to Auto, so it can be
                    // automatically sized depending on the text
                    bottom: Val::Auto,
                    left: Val::Auto,
                    // give it some padding for readability
                    padding: UiRect::all(Val::Px(4.0)),
                    ..Default::default()
                },
                ..Default::default()
            },
        ))
        .id();
    // create our text
    let text_fps = commands
        .spawn((
            FpsText,
            TextBundle {
                // use two sections, so it is easy to update just the number
                text: Text::from_sections([
                    TextSection {
                        value: "FPS: ".into(),
                        style: TextStyle {
                            font_size: 16.0,
                            color: Color::WHITE,
                            // if you want to use your game's font asset,
                            // uncomment this and provide the handle:
                            // font: my_font_handle
                            ..default()
                        },
                    },
                    TextSection {
                        value: " N/A".into(),
                        style: TextStyle {
                            font_size: 16.0,
                            color: Color::WHITE,
                            // if you want to use your game's font asset,
                            // uncomment this and provide the handle:
                            // font: my_font_handle
                            ..default()
                        },
                    },
                ]),
                ..Default::default()
            },
        ))
        .id();
    commands.entity(root).push_children(&[text_fps]);
}

fn fps_text_update_system(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut Text, With<FpsText>>,
) {
    for mut text in &mut query {
        // try to get a "smoothed" FPS value from Bevy
        if let Some(value) = diagnostics
            .get(FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|fps| fps.smoothed())
        {
            // Format the number as to leave space for 4 digits, just in case,
            // right-aligned and rounded. This helps readability when the
            // number changes rapidly.
            text.sections[1].value = format!("{value:>4.0}");

            // Let's make it extra fancy by changing the color of the
            // text according to the FPS value:
            text.sections[1].style.color = if value >= 120.0 {
                // Above 120 FPS, use green color
                Color::rgb(0.0, 1.0, 0.0)
            } else if value >= 60.0 {
                // Between 60-120 FPS, gradually transition from yellow to green
                Color::rgb((1.0 - (value - 60.0) / (120.0 - 60.0)) as f32, 1.0, 0.0)
            } else if value >= 30.0 {
                // Between 30-60 FPS, gradually transition from red to yellow
                Color::rgb(1.0, ((value - 30.0) / (60.0 - 30.0)) as f32, 0.0)
            } else {
                // Below 30 FPS, use red color
                Color::rgb(1.0, 0.0, 0.0)
            }
        } else {
            // display "N/A" if we can't get a FPS measurement
            // add an extra space to preserve alignment
            text.sections[1].value = " N/A".into();
            text.sections[1].style.color = Color::WHITE;
        }
    }
}

/// Toggle the FPS counter when pressing F12
fn fps_counter_showhide(mut q: Query<&mut Visibility, With<FpsRoot>>, kbd: Res<Input<KeyCode>>) {
    if kbd.just_pressed(KeyCode::F12) {
        let mut vis = q.single_mut();
        *vis = match *vis {
            Visibility::Hidden => Visibility::Visible,
            _ => Visibility::Hidden,
        };
    }
}

mod fabian;
mod maze_gen;
mod maze_render;
mod player_controller;
mod render;
mod shaders;
// mod terrain_loader;
mod terrain_render;
mod test_render;
mod tests;
mod tree_render;
mod terrain_loader;

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
    mut mats: ResMut<Assets<StandardMaterial>>,
    // mut commands: Commands,
    // mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut textures: ResMut<Assets<Image>>,
    // player_controller: Query<Entity>,
) {
    let terrain_gen = TerrainGenerator::new();
    let terrain_mesh = create_base_terrain_mesh();

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

    commands.spawn(Collider::heightfield(
        heights,
        dims,
        dims,
        Vec3::new(TILE_SIZE as f32 / 2., 1., TILE_SIZE as f32 / 2.),
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
                transform: Transform::from_xyz(0., 100., 0.),
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
                            // hdr: true,
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

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

fn main() {
    init_panic_hook();

    let mut app = App::new();
    app
        // .insert_resource(ClearColor(Color::rgb(0., 0., 0.)))
        // the sky color
        // .insert_resource(ClearColor(Color::rgb(0.5294, 0.8078, 0.9216)))
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
        // we do this last because the terrain rendering is attached to the player!
        .add_systems(Startup, create_terrain)
        // .add_systems(Startup, load_terrain)
        // .add_systems(RunFixedUpdateLoop, pan_orbit_camera)
        .add_systems(Update, (movement_input, mouse_look, toggle_cursor_lock));

    app.add_systems(Startup, setup_fps_counter);
    app.add_systems(Update, (fps_text_update_system, fps_counter_showhide));
    app.run();
    // print_render_graph(&mut app);
}
