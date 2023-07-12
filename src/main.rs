//! A simple 3D scene with light shining over a cube sitting on a plane.

mod maze_gen;

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::render::camera::Projection;
use bevy::window::PrimaryWindow;
use image::io::Reader as ImageReader;
use std::io::Cursor;
use image::{ImageBuffer, RgbImage};
use crate::maze_gen::{populate_maze, SquareMaze, SquareMazeComponent, SquareNode};

/// Tags an entity as capable of panning and orbiting.
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
    let translation = Vec3::new(-2.0, 2.5, 5.0);
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
        mesh: meshes.add(shape::Plane::from_size(10.0).into()),
        material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
        ..default()
    });
    let edited_cube = Mesh::from(shape::Box {
        min_x: -5.0,
        min_y: -0.5,
        min_z: -5.0,
        max_x: 0.0,
        max_y: 1.0,
        max_z: 5.0,
    });
    commands.spawn(PbrBundle {
        mesh: meshes.add(edited_cube),
        material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
        transform: Transform::from_xyz(0.0, 0.5, 0.0),
        ..default()
    });
    // cube
    // commands.spawn(PbrBundle {
    //     mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
    //     material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
    //     transform: Transform::from_xyz(0.0, 0.5, 0.0),
    //     ..default()
    // });
    // light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });
    // camera
    spawn_camera(commands)
}

fn main() {
    let node_count: u32 = 64;
    let mut starting_comp = SquareMazeComponent::new();
    starting_comp.add_node((0,0));
    let mut graph: SquareMaze = SquareMaze {
        maze: SquareMazeComponent::new(),
        size: node_count as i64,
        offset: (0, 0)
    };
    populate_maze(&mut graph, vec![starting_comp]);

    print!("{}\n", graph.maze.node_count().to_string());
    print!("{}", graph.maze.edge_count().to_string());

    let mut img: RgbImage = ImageBuffer::new(node_count * 2, node_count * 2);

    for x in 0..node_count as i64 {
        for y in 0..node_count as i64 {
            let chan_center = ((x * 2) as u32, (y * 2) as u32);
            img.put_pixel(chan_center.0, chan_center.1, image::Rgb([255, 255, 255]));
            if graph.maze.contains_edge((x, y), (x+1, y)) {
                img.put_pixel(chan_center.0 + 1, chan_center.1, image::Rgb([255, 255, 255]));
            }
            if graph.maze.contains_edge((x, y), (x, y+1)) {
                img.put_pixel(chan_center.0, chan_center.1 + 1, image::Rgb([255, 255, 255]));
            }
        }
    }

    img.save("maze_out.png").unwrap();
    // // Construct a new by repeated calls to the supplied closure.
    // let mut img = ImageBuffer::from_fn(512, 512, |x, y| {
    //     if x % 2 == 0 {
    //         image::Luma([0u8])
    //     } else {
    //         image::Luma([255u8])
    //     }
    // });

    // // Obtain the image's width and height.
    // let (width, height) = img.dimensions();
    //
    // // Access the pixel at coordinate (100, 100).
    // let pixel = img[(100, 100)];
    //
    // // Or use the `get_pixel` method from the `GenericImage` trait.
    // let pixel = *img.get_pixel(100, 100);
    //
    // // Put a pixel at coordinate (100, 100).
    // img.put_pixel(100, 100, pixel);
    //
    // // Iterate over all pixels in the image.
    // for pixel in img.pixels() {
    //     // Do something with pixel.
    // }

    // App::new()
    //     .add_plugins(DefaultPlugins)
    //     .add_startup_system(setup)
    //     .add_system(pan_orbit_camera)
    //     .run();
    //
    // // just to catch compilation errors
    // let _ = App::new().add_startup_system(spawn_camera);
}
