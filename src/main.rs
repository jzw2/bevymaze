//! A simple 3D scene with light shining over a cube sitting on a plane.

mod maze_gen;
mod maze_render;
mod test_render;

use crate::maze_gen::{
    populate_maze, CircleMaze, CircleMazeComponent, CircleNode, Maze, SquareMaze,
    SquareMazeComponent, SquareNode, SQUARE_MAZE_CELL_SIZE, SQUARE_MAZE_WALL_WIDTH,
};
use crate::maze_render::{
    get_arc_mesh, get_segment_mesh, polar_to_cart, Arc, Circle, GetWall, Segment,
};
use crate::test_render::{
    draw_circle, draw_segment, to_canvas_space, AxisTransform, DrawableCircle, DrawableSegment,
};
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::math::Vec3;
use bevy::prelude::*;
use bevy::render::camera::Projection;
use bevy::render::mesh::VertexAttributeValues;
use bevy::window::PrimaryWindow;
use image::io::Reader as ImageReader;
use image::{ImageBuffer, Rgb, RgbImage};
use imageproc::drawing::{draw_text, draw_text_mut};
use rand::{thread_rng, Rng, RngCore, SeedableRng};
use rusttype::{Font, Scale};
use std::f64::consts::PI;
use std::fmt::format;
use std::io::Cursor;

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
    mut query: Query<(&mut PanOrbitCamera, &mut Transform, &Projection)>,
) {
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
    } else if input_mouse.pressed(pan_button) {
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
            pan_orbit.radius -= scroll * pan_orbit.radius * 0.2;
            // dont allow zoom to reach zero or you get stuck
            pan_orbit.radius = f32::max(pan_orbit.radius, 0.05);
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
    let translation = Vec3::new(0.0, 2.5, -5.0);
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
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // plane
    commands.spawn(PbrBundle {
        mesh: meshes.add(shape::Plane::from_size(1000.0).into()),
        material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
        transform: Transform::from_xyz(0.0, -1.0, 0.0),
        ..default()
    });

    // let mut graph = CircleMaze {
    //     maze: CircleMazeComponent::new(),
    //     cell_size: 1.0,
    //     center: (0, 0),
    //     radius: 13,
    //     min_path_width: 1.0,
    //     wall_width: 0.1
    // };
    // let mut starting_comp = CircleMazeComponent::new();
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

    for mesh in graph.get_wall_geometry(0.1, 0.4) {
        commands.spawn(PbrBundle {
            mesh: meshes.add(mesh),
            material: materials.add(StandardMaterial {
                base_color: Color::rgba(0.01, 0.8, 0.2, 1.0).into(),
                alpha_mode: AlphaMode::Multiply,
                ..default()
            }),
            transform: Transform::from_xyz(0.0, 0.0, 0.0),
            ..default()
        });
    }

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
    commands.insert_resource(ClearColor(Color::rgb_u8(135, 206, 235)));
    // camera
    spawn_camera(commands)
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup)
        .add_system(pan_orbit_camera)
        .run();

    // just to catch compilation errors
    let _ = App::new().add_startup_system(spawn_camera);
}
