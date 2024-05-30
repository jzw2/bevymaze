use crate::fabian::find_triangle;
use crate::loader::ServerRequestSender;
use crate::maze_loader::MazeResponseReceiver;
use crate::player_controller::{PlayerBody, PlayerCam};
use crate::shaders::{TerrainMaterial, TerrainMaterialDataHolder, MAX_VERTICES};
use crate::terrain_render::*;
use bevy::asset::AssetsMutIterator;
use bevy::ecs::system::{CommandQueue, SystemState};
use bevy::log::Level;
use bevy::math::{DVec2, DVec3};
use bevy::prelude::*;
use bevy::reflect::erased_serde::__private::serde::{Deserialize, Serialize};
use bevy::reflect::List;
use bevy::render::mesh::VertexAttributeValues::Float32x3;
use bevy::render::render_resource::encase::private::RuntimeSizedArray;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::tasks::{block_on, AsyncComputeTaskPool, Task};
use bevy_flycam::FlyCam;
use bevy_rapier3d::geometry::Collider;
use bevy_tokio_tasks::TokioTasksRuntime;
use crossbeam_channel::{bounded, Receiver, Sender};
use delaunator::{triangulate, Point, Triangulation};
use futures_lite::{future, StreamExt};
use futures_util::SinkExt;
use hilbert::transform;
use itertools::{enumerate, Itertools};
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use postcard::{from_bytes, to_stdvec, to_vec};
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use server::connection::MazeNetworkResponse::TerrainHeights;
use server::connection::*;
use server::terrain_gen::TILE_SIZE;
use server::util::{barycentric32, lin_map};
use std::mem::size_of;
use std::net::{Ipv4Addr, SocketAddr};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Error;
use tokio_tungstenite::tungstenite::Message::Binary;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use url::Url;
use crate::terrain_data_map::TerrainDataMap;

/// Quarter meter data tolerance when the player is 1m away
const DATA_TOLERANCE: f64 = 0.5;

#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct MainTerrainColldier;

pub struct TerrainDataUpdate {
    guesses: Vec<u32>,
    vertices: Vec<f32>,
    triangles: Vec<u32>,
    halfedges: Vec<u32>,
    heights: Vec<f32>,
    gradients: Vec<f32>,
    center: Vec2,
    collider: Collider,
}

#[derive(Resource, Deref)]
pub struct TerrainResponseSender(pub Sender<Vec<TerrainDataPoint>>);

#[derive(Resource, Deref)]
pub struct TerrainResponseReceiver(pub Receiver<Vec<TerrainDataPoint>>);

#[derive(Resource, Deref)]
pub struct TerrainUpdateReceiver(Receiver<TerrainDataUpdate>);

#[derive(Resource)]
pub struct MeshOffsetRes(pub Arc<Mutex<GlobalTransform>>);

#[derive(Resource)]
pub struct TerrainTransformMaterialRes(pub Handle<TerrainMaterial>);

#[derive(Resource)]
pub struct TerrainUpdateProcHandle(pub JoinHandle<()>);

pub fn setup_transform_res(
    mut commands: Commands,
    transform: Query<&GlobalTransform, (With<PlayerCam>, Without<PlayerBody>)>,
    // transform: Query<&GlobalTransform, (With<FlyCam>, Without<PlayerBody>)>,
) {
    println!("Setup transform res {}", transform.is_empty());
    let tf = transform.single();
    let mutex = Mutex::new(tf.clone());
    let pointer = Arc::new(mutex);
    commands.insert_resource(MeshOffsetRes(pointer));
}

/// TODO: Verify this does what it should 🤷
pub fn update_transform_res(
    transform_res: ResMut<MeshOffsetRes>,
    transform: Query<&GlobalTransform, (With<PlayerCam>, Without<PlayerBody>)>,
    // transform: Query<&GlobalTransform, (With<FlyCam>, Without<PlayerBody>)>,
    mut terrain_material: ResMut<Assets<TerrainMaterial>>,
) {
    let mut old = (transform.single().translation().clone() / LATTICE_GRID_SIZE as f32).round()
        * LATTICE_GRID_SIZE as f32;
    // old.x = old.x.round();
    // old.z = old.z.round();
    let new_transform = GlobalTransform::from_xyz(old.x, 0., old.z);
    {
        let mut val = transform_res.0.lock().unwrap();
        *val = new_transform;
    }
    // // also update the shader
    if let Some((handle, material)) = terrain_material.iter_mut().next() {
        material.transform = old.xz();
    }
}

pub fn setup_terrain_loader(
    mut commands: Commands,
    runtime: ResMut<TokioTasksRuntime>,
    transform: Res<MeshOffsetRes>,
    terrain_data_map: Res<TerrainDataMap>,
    mut terrain_material: ResMut<Assets<TerrainMaterial>>,
    mut terrain_material_data_holder: ResMut<TerrainMaterialDataHolder>,
    render_device: ResMut<RenderDevice>,
    render_queue: ResMut<RenderQueue>,
    server_request_sender: ResMut<ServerRequestSender>,
    terrain_response_receiver: ResMut<TerrainResponseReceiver>,
) {
    let (handle, material) = terrain_material.iter_mut().next().unwrap();
    // setup the shader verts!

    terrain_material_data_holder.mesh_vertices.clear();
    terrain_material_data_holder
        .mesh_vertices
        .values_mut()
        .append(
            &mut terrain_data_map
                .mesh
                .terrain_mesh_verts
                .clone()
                .into_iter()
                .flatten()
                .collect(),
        );

    terrain_material_data_holder
        .mesh_vertices
        .write_buffer(&*render_device, &*render_queue);

    let (tx, rx) = bounded::<TerrainDataUpdate>(5);
    let transform = Arc::clone(&transform.0);

    let socket = server_request_sender.0.clone();
    let terrain_res_rec = terrain_response_receiver.0.clone();

    // we just use this once, and it lives ~forever~ (spooky)
    let mut map = terrain_data_map.clone();
    let mesh_verts = terrain_data_map.mesh.terrain_mesh_verts.clone();
    commands.insert_resource::<TerrainUpdateProcHandle>(TerrainUpdateProcHandle(
        runtime.spawn_background_task(|_ctx| async move {
            loop {
                // update the center
                {
                    let t = transform.lock().unwrap();
                    map.mesh.center = t.translation().xz();
                    // println!("Trans {:?}", map.center);
                }

                let batch_size = 1024; //512 / size_of::<TerrainDataPoint>();
                let mut to_fetch = Vec::<TerrainDataPoint>::with_capacity(batch_size);
                let mut total = 0;
                for i in 0..mesh_verts.len() {
                    // now we check the data map
                    if let Some(vert) = map.vert_to_fetch(i) {
                        to_fetch.push(TerrainDataPoint {
                            coordinates: vert,
                            height: 0.,
                            idx: 0,
                            gradient: [0., 0.],
                        });
                    }

                    // send off a load and mark requested
                    // we do this so the server can start fetching data for us
                    if to_fetch.len() >= batch_size
                        || (i == mesh_verts.len() - 1 && to_fetch.len() > 0)
                    {
                        total += to_fetch.len();
                        println!("Sending req for {} {total}", to_fetch.len());
                        map.mark_requested(&mut to_fetch);
                        socket
                            .send(MazeNetworkRequest::ReqTerrainHeights(to_fetch.clone()))
                            .expect("TODO: panic message");
                        to_fetch.clear();
                    }
                }

                if total == 0 {
                    continue;
                }

                while total > 0 {
                    if let Some(fetched) = terrain_res_rec.iter().next() {
                        map.fill_in_data(&fetched);
                        println!(
                            "Filled in {}, remaining {}",
                            fetched.len(),
                            total - fetched.len()
                        );
                        total -= fetched.len();
                    }
                }

                // finally update!

                // send a signal to update our shader once we've received everything we requested
                let mut delaunay_points: Vec<Point> = Vec::with_capacity(MAX_VERTICES);
                let mut raw: Vec<f32> = Vec::with_capacity(MAX_VERTICES * 2);
                let mut heights: Vec<f32> = Vec::with_capacity(MAX_VERTICES);
                let mut gradients: Vec<f32> = Vec::with_capacity(MAX_VERTICES * 2);
                let mut guesses = vec![0u32; mesh_verts.len()];

                let mut verts = Vec::<[f32; 2]>::with_capacity(mesh_verts.len());

                for datum in &map.resolved_data.data {
                    // generate a random position in a circle
                    let [x, z] = datum.coordinates;

                    if x.is_infinite() || z.is_infinite() {
                        // ignore unresolved data
                        continue;
                    }

                    let y = datum.height;
                    delaunay_points.push(Point {
                        x: x as f64,
                        y: z as f64,
                    });
                    heights.push(y);
                    raw.push(x);
                    raw.push(z);
                    gradients.push(datum.gradient[0]);
                    gradients.push(datum.gradient[1]);
                    verts.push([x, z]);
                }

                let triangulation = triangulate(&delaunay_points);

                for (i, vtx) in map.mesh.terrain_mesh_verts.iter_mut().enumerate() {
                    let mut vtx = *vtx;
                    vtx[0] += map.mesh.center.x;
                    vtx[1] += map.mesh.center.y;
                    let prev = {
                        if i < 1 {
                            0
                        } else {
                            guesses[i - 1]
                        }
                    };
                    guesses[i] =
                        find_triangle(&verts, &triangulation, &vtx, prev as usize, usize::MAX)
                            as u32;
                }

                println!(
                    "Sending data r {} | v {} t {} he {} h {} g {}",
                    map.resolved_data.data.len(),
                    raw.len(),
                    triangulation.triangles.len(),
                    triangulation.halfedges.len(),
                    heights.len(),
                    gradients.len()
                );
                tx.send(TerrainDataUpdate {
                    center: map.mesh.center,
                    collider: generate_terrain_collider(
                        &map.mesh.center,
                        &verts,
                        &heights,
                        &gradients,
                        &triangulation,
                    ),
                    vertices: raw,
                    triangles: triangulation
                        .triangles
                        .into_iter()
                        .map(|e| e as u32)
                        .collect(),
                    halfedges: triangulation
                        .halfedges
                        .into_iter()
                        .map(|e| e as u32)
                        .collect(),
                    heights,
                    gradients,
                    guesses,
                })
                .unwrap();
            }
        }),
    ));
    commands.insert_resource(TerrainUpdateReceiver(rx));
}

fn generate_terrain_collider(
    center: &Vec2,
    verts: &Vec<[f32; 2]>,
    heights: &Vec<f32>,
    gradients: &Vec<f32>,
    triangulation: &Triangulation,
) -> Collider {
    // we generate a heightfield from the
    let mut collider_heights: Vec<f32> = vec![];
    let dims = 100;
    let mut last_guess = 0;
    let apothem = TILE_SIZE / 16.;
    for x in 0..dims {
        for z in 0..dims {
            let xp = lin_map(
                0.,
                dims as f64 - 1.,
                -apothem + center.x as f64,
                apothem + center.x as f64,
                x as f64,
            ) as f32;
            let zp = lin_map(
                0.,
                dims as f64 - 1.,
                -apothem + center.y as f64,
                apothem + center.y as f64,
                z as f64,
            ) as f32;
            let p = [xp, zp];
            // get the triangle
            last_guess = find_triangle(&verts, &triangulation, &p, last_guess, usize::MAX);
            let t = (last_guess / 3) * 3;
            // INTERPOLATE!
            let (A, B, C) = (
                triangulation.triangles[t],
                triangulation.triangles[t + 1],
                triangulation.triangles[t + 2],
            );
            let (a, b) = barycentric32(&p, &[verts[A], verts[B], verts[C]]);
            collider_heights.push(bary_poly_smooth_interp(A, B, C, heights, gradients, [a, b]));
        }
    }

    Collider::heightfield(
        collider_heights,
        dims,
        dims,
        Vec3::new(apothem as f32 * 2., 1.0, apothem as f32 * 2.),
    )
}

pub fn stream_terrain_mesh(
    mut commands: Commands,
    terrain_collider: Query<Entity, With<MainTerrainColldier>>,
    receiver: Res<TerrainUpdateReceiver>,
    mut terrain_material: ResMut<Assets<TerrainMaterial>>,
    mut terrain_material_data_holder: ResMut<TerrainMaterialDataHolder>,
    mut render_device: ResMut<RenderDevice>,
    mut render_queue: ResMut<RenderQueue>,
) {
    let last = receiver.try_iter().last();
    if let Some(update) = last {
        // first update the collider
        let collider = terrain_collider.single();
        commands.entity(collider).despawn();
        commands.spawn((
            update.collider,
            MainTerrainColldier,
            TransformBundle {
                local: Transform::from_translation(Vec3::new(update.center.x, 0., update.center.y)),
                ..default()
            },
        ));

        terrain_material_data_holder.vertices.clear();
        terrain_material_data_holder
            .vertices
            .values_mut()
            .append(&mut update.vertices.clone());

        terrain_material_data_holder.triangles.clear();
        terrain_material_data_holder
            .triangles
            .values_mut()
            .append(&mut update.triangles.clone());

        terrain_material_data_holder.halfedges.clear();
        terrain_material_data_holder
            .halfedges
            .values_mut()
            .append(&mut update.halfedges.clone());

        terrain_material_data_holder.height.clear();
        terrain_material_data_holder
            .height
            .values_mut()
            .append(&mut update.heights.clone());

        terrain_material_data_holder.gradients.clear();
        terrain_material_data_holder
            .gradients
            .values_mut()
            .append(&mut update.gradients.clone());

        terrain_material_data_holder.triangle_indices.clear();
        terrain_material_data_holder
            .triangle_indices
            .values_mut()
            .append(&mut update.guesses.clone());

        terrain_material_data_holder.write_buffer(&*render_device, &*render_queue);

        let (handle, material) = terrain_material.iter_mut().next().unwrap();
        material.triangles = terrain_material_data_holder
            .triangles
            .buffer()
            .unwrap()
            .clone();
        material.halfedges = terrain_material_data_holder
            .halfedges
            .buffer()
            .unwrap()
            .clone();
        material.vertices = terrain_material_data_holder
            .vertices
            .buffer()
            .unwrap()
            .clone();
        material.height = terrain_material_data_holder
            .height
            .buffer()
            .unwrap()
            .clone();
        material.triangle_indices = terrain_material_data_holder
            .triangle_indices
            .buffer()
            .unwrap()
            .clone();
        material.gradients = terrain_material_data_holder
            .gradients
            .buffer()
            .unwrap()
            .clone();
        material.mesh_vertices = terrain_material_data_holder
            .mesh_vertices
            .buffer()
            .unwrap()
            .clone();

        // material.triangle_indices = update.guesses;
    }
}