use crate::fabian::find_triangle;
use crate::player_controller::{PlayerBody, PlayerCam};
use crate::shaders::{TerrainMaterial, MAX_VERTICES};
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
use bevy::tasks::{block_on, AsyncComputeTaskPool, Task};
use bevy_rapier3d::geometry::Collider;
use bevy_tokio_tasks::TokioTasksRuntime;
use crossbeam_channel::{bounded, Receiver};
use delaunator::{triangulate, Point, Triangulation};
use futures_lite::{future, StreamExt};
use futures_util::SinkExt;
use itertools::{enumerate, Itertools};
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use lightyear::prelude::client::*;
use lightyear::prelude::{
    ClientId, Io, IoConfig, LinkConditionerConfig, LogConfig, Message, PingConfig, SharedConfig,
    TickConfig, TransportConfig,
};
use postcard::{from_bytes, to_stdvec, to_vec};
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use server::connection::*;
use server::terrain_gen::TILE_SIZE;
use server::util::{barycentric32, lin_map};
use std::mem::size_of;
use std::net::{Ipv4Addr, SocketAddr};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use kiddo::float::kdtree::KdTree;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Error;
use tokio_tungstenite::tungstenite::Message::Binary;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use url::Url;

/// Quarter meter data tolerance when the player is 1m away
const DATA_TOLERANCE: f64 = 0.5;

#[derive(Resource, Clone, Copy)]
pub struct MyClientPlugin {
    pub(crate) client_id: ClientId,
    pub(crate) client_port: u16,
    pub(crate) server_addr: Ipv4Addr,
    pub(crate) server_port: u16,
    pub(crate) transport: Transports,
}

impl Plugin for MyClientPlugin {
    fn build(&self, app: &mut App) {
        let server_addr = SocketAddr::new(self.server_addr.into(), self.server_port);
        let auth = Authentication::Manual {
            server_addr,
            client_id: self.client_id,
            private_key: KEY,
            protocol_id: PROTOCOL_ID,
        };
        let client_addr = SocketAddr::new(Ipv4Addr::UNSPECIFIED.into(), self.client_port);
        let link_conditioner = LinkConditionerConfig {
            incoming_latency: Duration::from_millis(200),
            incoming_jitter: Duration::from_millis(20),
            incoming_loss: 0.05,
        };
        let transport = match self.transport {
            Transports::Udp => TransportConfig::UdpSocket(client_addr),
            Transports::Webtransport => TransportConfig::WebTransportClient {
                client_addr,
                server_addr,
            },
        };
        let io =
            Io::from_config(IoConfig::from_transport(transport).with_conditioner(link_conditioner));
        let config = ClientConfig {
            shared: shared_config().clone(),
            input: InputConfig::default(),
            netcode: Default::default(),
            ping: PingConfig::default(),
            sync: SyncConfig::default(),
            prediction: PredictionConfig::default(),
            // we are sending updates every frame (60fps), let's add a delay of 6 network-ticks
            interpolation: InterpolationConfig::default()
                .with_delay(InterpolationDelay::default().with_send_interval_ratio(2.0)),
        };
        let plugin_config = PluginConfig::new(config, io, protocol(), auth);
        app.add_plugins(ClientPlugin::new(plugin_config));
        // app.add_plugins(crate::shared::SharedPlugin);
        app.insert_resource(self.clone());
        app.add_systems(Startup, init);
        // app.add_systems(
        //     FixedUpdate,
        //     buffer_input.in_set(InputSystemSet::BufferInputs),
        // );
        // app.add_systems(FixedUpdate, player_movement.in_set(FixedUpdateSet::Main));
        // app.add_systems(
        //     Update,
        //     (
        //         receive_message1,
        //         receive_entity_spawn,
        //         receive_entity_despawn,
        //         handle_predicted_spawn,
        //         handle_interpolated_spawn,
        //     ),
        // );
    }
}

// Startup system for the client
pub(crate) fn init(
    mut commands: Commands,
    mut client: ResMut<Client>,
    plugin: Res<MyClientPlugin>,
) {
    client.connect();
}

#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct MainTerrainColldier;

pub struct TerrainDataUpdate {
    guesses: Vec<u32>,
    vertices: Vec<f32>,
    triangules: Vec<u32>,
    halfedges: Vec<u32>,
    heights: Vec<f32>,
    gradients: Vec<f32>,
    center: Vec2,
    collider: Collider,
}

#[derive(Resource, Deref)]
pub struct TerrainUpdateReceiver(Receiver<TerrainDataUpdate>);

#[derive(Resource)]
pub struct MeshOffsetRes(Arc<Mutex<GlobalTransform>>);

pub fn setup_transform_res(
    mut commands: Commands,
    transform: Query<&GlobalTransform, (With<PlayerCam>, Without<PlayerBody>)>,
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
) {
    let mut val = transform_res.0.lock().unwrap();
    *val = *transform.single();
}

pub fn setup_terrain_loader(
    mut commands: Commands,
    runtime: ResMut<TokioTasksRuntime>,
    transform: Res<MeshOffsetRes>,
    terrain_data_map: Res<TerrainDataMap>,
) {
    let (tx, rx) = bounded::<TerrainDataUpdate>(5);
    let transform = Arc::clone(&transform.0);

    // we just use this once, and it lives ~forever~ (spooky)
    let mut map = terrain_data_map.clone();
    let mesh_verts = terrain_data_map.mesh.terrain_mesh_verts.clone();
    runtime.spawn_background_task(|_ctx| async move {
        // now fetch from the server!
        let (mut socket, other) =
            connect_async(Url::parse("ws://127.0.0.1:9002").expect("Can't connect to URL"))
                .await
                .unwrap();

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
                if to_fetch.len() >= batch_size || (i == mesh_verts.len() - 1 && to_fetch.len() > 0)
                {
                    total += to_fetch.len();
                    // println!("Sending req for {}", to_fetch.len());
                    map.mark_requested(&mut to_fetch);
                    socket
                        .send(Binary(to_stdvec(&to_fetch).unwrap()))
                        .await
                        .expect("TODO: panic message");
                    // now clear and restart
                    to_fetch.clear();
                }
            }

            if total == 0 {
                continue;
            }

            while total > 0 {
                match socket.next().await.unwrap() {
                    Ok(msg) => {
                        if let Binary(bin) = msg {
                            let fetched = from_bytes::<Vec<TerrainDataPoint>>(&bin).unwrap();
                            map.fill_in_data(&fetched);
                            println!(
                                "Filled in {}, remaining {}",
                                fetched.len(),
                                total - fetched.len()
                            );
                            total -= fetched.len();
                        }
                    }
                    Err(_) => {
                        // do nothing
                    }
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
                    find_triangle(&verts, &triangulation, &vtx, prev as usize, usize::MAX) as u32;
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
                triangules: triangulation
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
    });
    // tokio::spawn();

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
            collider_heights.push(bary_poly_smooth_interp(
                A,
                B,
                C,
                heights,
                gradients,
                [a, b],
            ));
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

        let (handle, material) = terrain_material.iter_mut().next().unwrap();
        material.vertices = update.vertices;
        material.triangles = update.triangules;
        material.halfedges = update.halfedges;
        material.height = update.heights;
        material.gradients = update.gradients;
        material.triangle_indices = update.guesses;
    }
}

#[derive(Clone)]
pub struct TerrainDataHolder {
    /// The actual data we have on hand
    /// The amount of data is length 2N. The last N elements are dedicated to elements
    /// that have we have not fetched yet
    data: Vec<TerrainDataPoint>,
    /// The locator lets us find which point in the terrain data we are closest to quickly
    /// This also includes unresolved data
    locator: KdTree<f32, u64, 2, 128, u32>,
    /// This is the list of eviction candidates from best to worst
    /// It is sorted by the candidates' ratio of actual distance to ideal distance
    eviction_priority: Vec<(usize, f32)>,
}

#[derive(Clone)]
pub struct TerrainMeshHolder {
    /// An unordered list of vertices in the mesh
    terrain_mesh_verts: Vec<[f32; 2]>,
    /// The reverse locator lets us find which point in the terrain MESH we are closest to quickly
    reverse_locator: KdTree<f32, u64, 2, 32, u32>,
    /// Center of the terrain mesh
    center: Vec2,
    /// The radius of acceptance for each vertex in terrain_mesh_verts
    check_radii: Vec<f32>,
}

#[derive(Resource, Clone)]
pub struct TerrainDataMap {
    initialized: bool,
    resolved_data: TerrainDataHolder,
    unresolved_data: TerrainDataHolder,
    mesh: TerrainMeshHolder,
}

/// How much the nodes are rounded by
const ACCURACY: f32 = 0.25;

impl TerrainDataHolder {
    /// Partially sort the priorities by the first n highest
    pub fn select_highest_priorities(&mut self, mesh: &TerrainMeshHolder, n_highest: usize) {
        for (i, data) in (&self.data).into_iter().enumerate() {
            if data.coordinates[0].is_infinite() || data.coordinates[1].is_infinite() {
                self.eviction_priority[i] = (i, f32::INFINITY);
            } else {
                // correct for coordinate being in world space, we want it relative to the mesh
                let nearest = mesh.reverse_locator.nearest_one::<SquaredEuclidean>(&[
                    data.coordinates[0] - mesh.center.x,
                    data.coordinates[1] - mesh.center.y,
                ]);
                let required_dist = mesh.get_check_radius(nearest.item as usize);
                self.eviction_priority[i] = (i, nearest.distance / required_dist);
            }
        }
        self.eviction_priority
            .select_nth_unstable_by(n_highest - 1, |a, b| b.1.total_cmp(&a.1));
    }
}

impl TerrainMeshHolder {
    /// Get the radius of acceptable data for a particular vertex in the terrain mesh
    pub fn get_check_radius(&self, vertex_idx: usize) -> f32 {
        return self.check_radii[vertex_idx];
    }
}

impl TerrainDataMap {
    pub fn new(terrain_lattice: &Vec<DVec3>) -> Self {
        return TerrainDataMap::new_with_capacity(terrain_lattice, MAX_VERTICES);
    }

    pub fn new_with_capacity(terrain_lattice: &Vec<DVec3>, capacity: usize) -> Self {
        let lattice_data: Vec<[f32; 2]> = terrain_lattice
            .into_iter()
            .map(|e| e.xz().as_vec2().to_array())
            .collect();
        let holder = TerrainDataHolder {
            data: vec![
                TerrainDataPoint {
                    coordinates: [f32::INFINITY, f32::INFINITY],
                    height: 0.0,
                    idx: 0,
                    gradient: [0., 0.]
                };
                capacity
            ],
            locator: Default::default(),
            eviction_priority: Vec::from_iter(
                vec![f32::INFINITY; capacity].into_iter().enumerate(),
            ),
        };
        let check_radii = terrain_lattice
            .into_iter()
            .map(|e| (DATA_TOLERANCE * e.xz().length()) as f32)
            .collect();
        return TerrainDataMap {
            initialized: false,
            resolved_data: holder.clone(),
            unresolved_data: holder,
            mesh: TerrainMeshHolder {
                reverse_locator: KdTree::<f32, u64, 2, 32, u32>::from(&lattice_data),
                terrain_mesh_verts: lattice_data,
                center: Vec2::ZERO,
                check_radii,
            },
        };
    }

    /// Mark these vertices as requested data
    /// We potentially evict other requested data before it's been filled in
    pub fn mark_requested(&mut self, data: &mut Vec<TerrainDataPoint>) {
        // add it to the locator to mark it as part of the useful data
        self.unresolved_data
            .select_highest_priorities(&self.mesh, data.len());
        for (i, datum) in data.into_iter().enumerate() {
            // evict any old unresolved data
            let (evicting, _) = self.unresolved_data.eviction_priority[i];
            let old_data = &self.unresolved_data.data[evicting];
            self.unresolved_data
                .locator
                .remove(&old_data.coordinates, evicting as u64);
            datum.idx = evicting;
            self.unresolved_data.data[evicting] = datum.clone();
            self.unresolved_data
                .locator
                .add(&datum.coordinates, datum.idx as u64);
        }
    }

    pub fn fill_in_data(&mut self, data: &Vec<TerrainDataPoint>) -> bool {
        // first check the requested data and make sure this point still exists
        let data = data
            .iter()
            .filter(|new_data| {
                let old_data = &mut self.unresolved_data.data[new_data.idx];
                return old_data.coordinates[0] == new_data.coordinates[0]
                    && old_data.coordinates[1] == new_data.coordinates[1];
            })
            .collect_vec();
        // prioritize for replacement
        self.resolved_data
            .select_highest_priorities(&self.mesh, data.len());
        let empty = data.is_empty();
        // finally replace
        for (i, new_data) in data.into_iter().enumerate() {
            // first check the requested data and make sure this point still exists
            let mut old_unresolved = &mut self.unresolved_data.data[new_data.idx];
            // it does match, so remove this unresolved data
            self.unresolved_data
                .locator
                .remove(&old_unresolved.coordinates, old_unresolved.idx as u64);
            old_unresolved.coordinates = [f32::INFINITY, f32::INFINITY];

            // now replace the resolved data
            let (evicting, _) = self.resolved_data.eviction_priority[i];
            // println!(
            //     "Filling in {:?}, evicting {}",
            //     new_data.coordinates, evicting
            // );
            let old_resolved = &mut self.resolved_data.data[evicting];
            self.resolved_data
                .locator
                .remove(&old_resolved.coordinates, old_resolved.idx as u64);
            old_resolved.coordinates = new_data.coordinates;
            old_resolved.height = new_data.height;
            old_resolved.gradient = new_data.gradient;
            old_resolved.idx = evicting;
            self.resolved_data
                .locator
                .add(&old_resolved.coordinates, old_resolved.idx as u64);
        }

        return !empty;
    }

    /// Checks whether the terrain data is still acceptable to interpolate the specified point
    /// Data further from the camera has a higher tolerance for being off
    pub fn vert_to_fetch(&self, vertex_idx: usize) -> Option<[f32; 2]> {
        let mut vertex = self.mesh.terrain_mesh_verts[vertex_idx];
        vertex[0] += self.mesh.center.x;
        vertex[1] += self.mesh.center.y;
        // we check if the nearest guy is in the acceptable range
        let nearest = self
            .resolved_data
            .locator
            .nearest_one::<SquaredEuclidean>(&vertex);
        let unresolved_nearest = self
            .unresolved_data
            .locator
            .nearest_one::<SquaredEuclidean>(&vertex);
        if nearest.distance.min(unresolved_nearest.distance)
            > self.mesh.get_check_radius(vertex_idx)
        {
            return Some(vertex);
        }
        return None;
    }
}

#[test]
fn fill_in_when_full_test() {
    let mut rng = StdRng::seed_from_u64(1);
    const MESH_VERTS: usize = 80;
    const TERRAIN_VERTS: usize = MESH_VERTS * 5 / 2;
    let lattice = create_base_lattice_with_verts(MESH_VERTS as f64);
    let mesh_verts = lattice.len();
    let mut map = TerrainDataMap::new_with_capacity(&lattice, TERRAIN_VERTS);
    // completely fill up the map
    let mut data = Vec::<TerrainDataPoint>::with_capacity(TERRAIN_VERTS);
    for _ in 0..TERRAIN_VERTS {
        data.push(TerrainDataPoint {
            coordinates: [
                rng.gen_range(-X_VIEW_DIST_M..X_VIEW_DIST_M) as f32,
                rng.gen_range(-Z_VIEW_DIST_M..Z_VIEW_DIST_M) as f32,
            ],
            height: 0.,
            gradient: [0., 0.],
            idx: 0,
        });
    }
    map.mark_requested(&mut data);
    map.fill_in_data(&data);

    // now try to add new data multiple times
    for _ in 0..100 {
        // set a mew center
        map.mesh.center = Vec2::new(
            rng.gen_range(-10000.0..10000.0),
            rng.gen_range(-10000.0..10000.0),
        );
        const BATCH_SIZE: usize = 10;
        let mut total = 0;
        let mut batches = Vec::<Vec<TerrainDataPoint>>::with_capacity(BATCH_SIZE);
        let mut to_fetch = Vec::<TerrainDataPoint>::with_capacity(BATCH_SIZE);
        for i in 0..mesh_verts {
            if let Some(vert) = map.vert_to_fetch(i) {
                println!("- - - -Fetching {} {:?}", i, vert);
                to_fetch.push(TerrainDataPoint {
                    coordinates: vert,
                    height: 0.,
                    idx: 0,
                    gradient: [0., 0.],
                });
            }

            // send off a load and mark requested
            // we do this so the server can start fetching data for us
            if to_fetch.len() >= BATCH_SIZE || (i == mesh_verts - 1 && to_fetch.len() > 0) {
                total += to_fetch.len();
                // println!("Sending req for {}", to_fetch.len());
                map.mark_requested(&mut to_fetch);
                batches.push(to_fetch.clone());
                // now clear and restart
                to_fetch.clear();
            }
        }

        while total > 0 {
            let fetched = batches.pop().unwrap();
            map.fill_in_data(&fetched);
            println!(
                "Filled in {}, remaining {}",
                fetched.len(),
                total - fetched.len()
            );
            total -= fetched.len();
        }

        let center = [map.mesh.center.x, map.mesh.center.y];

        // now check that all these guys are in there
        for i in 0..map.mesh.terrain_mesh_verts.len() {
            let mut v = map.mesh.terrain_mesh_verts[i];
            v[0] += center[0];
            v[1] += center[1];
            let dist = map
                .resolved_data
                .locator
                .nearest_one::<SquaredEuclidean>(&v);
            println!(
                "Checking {} {:?} D {} CR {}",
                i,
                v,
                dist.distance,
                map.mesh.get_check_radius(i)
            );
            assert!(dist.distance < map.mesh.get_check_radius(i));
        }
    }
}
