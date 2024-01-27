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
use crossbeam_channel::{bounded, Receiver};
use delaunator::{triangulate, Point};
use futures_lite::future;
use futures_util::SinkExt;
use itertools::{enumerate, Itertools};
use kiddo::{ImmutableKdTree, KdTree, SquaredEuclidean};
use lightyear::prelude::client::*;
use lightyear::prelude::{
    ClientId, Io, IoConfig, LinkConditionerConfig, LogConfig, Message, PingConfig, SharedConfig,
    TickConfig, TransportConfig,
};
use postcard::{to_stdvec, to_vec};
use rand::Rng;
use server::connection::*;
use std::mem::size_of;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message::Binary;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use url::Url;

/// Quarter meter data tolerance when the player is 1m away
const DATA_TOLERANCE: f32 = 0.25;

#[derive(Resource)]
struct TerrainData(KdTree<f64, 2>);

fn get_check_radius(orig: &Vec2, center: &Vec2) -> f32 {
    let actual_pos = *orig - *center;
    let origin_dist = actual_pos.length();
    return origin_dist * DATA_TOLERANCE;
}

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
        app.insert_resource(VtxFetchIter(0));
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

#[derive(Resource)]
pub struct VtxFetchIter(usize);

const TO_FETCH: usize = 1;

#[derive(Component)]
pub struct ComputeRequiredData(Task<Vec<TerrainDataPoint>>);

#[derive(Component)]
pub struct FillInData(Task<(bool, TerrainDataMap)>);

pub fn request_terrain_vertices(
    mut commands: Commands,
    terrain_transform: Query<
        &mut Transform,
        (With<MainTerrain>, Without<PlayerBody>, Without<PlayerCam>),
    >,
    terrain_mesh: Query<&Handle<Mesh>, With<MainTerrain>>,
    meshes: Res<Assets<Mesh>>,
    mut terrain_data_map: ResMut<TerrainDataMap>,
    mut client: ResMut<Client>,
    mut fetch_iter: ResMut<VtxFetchIter>,
    mut check_tasks: Query<(Entity, &mut ComputeRequiredData)>,
) {
    // task priority:
    // 1. fill in data
    // 2. fetch data
    // 3. compute required data

    let thread_pool = AsyncComputeTaskPool::get();

    // make sure there's not already a check task running
    if check_tasks.is_empty() {
        // make sure there's not unresolved data still being waited on.
        // this is equivalent to waiting on the data to be fetched and filled
        if terrain_data_map.unresolved_data.locator.size() == 0 {
            // we get which vertices need to be fetched
            let terrain_handle = terrain_mesh.get_single().unwrap();
            let terrain = meshes.get(terrain_handle).unwrap();
            let terrain_tf = terrain_transform.get_single().unwrap();

            println!(
                "Terrain tf {} {}",
                terrain_tf.translation.x, terrain_tf.translation.z
            );
            terrain_data_map.center = terrain_tf.translation.xz();

            if let Some(Float32x3(verts)) = terrain.attribute(Mesh::ATTRIBUTE_POSITION) {
                let entity = commands.spawn_empty().id();
                let data = verts.clone();
                let data_map = terrain_data_map.clone();

                let task = thread_pool.spawn(async move {
                    let mut to_fetch = Vec::<TerrainDataPoint>::new();
                    for vert in &data {
                        let [x, y, z] = vert;
                        // now we check the data map
                        if data_map.should_fetch_more(&[*x, *z]) {
                            to_fetch.push(TerrainDataPoint {
                                coordinates: [*x, *z],
                                height: 0.,
                                idx: 0,
                                gradient: [0., 0.],
                            });
                        }
                    }

                    to_fetch
                });
                commands.entity(entity).insert(ComputeRequiredData(task));
            }
        }
    } else {
        // it isn't empty, meaning there's potentially some data that we can start fetching
        for (entity, mut task) in &mut check_tasks {
            let poll: Option<Vec<TerrainDataPoint>> = block_on(future::poll_once(&mut task.0));
            if let Some(mut to_fetch) = poll {
                // we have the data, so make a request
                if !to_fetch.is_empty() {
                    const CAP: usize = 1024;
                    let iters = to_fetch.len().div_ceil(CAP);

                    terrain_data_map.mark_requested(&mut to_fetch);

                    for i in 0..iters {
                        let start = i * CAP;
                        let end = ((i + 1) * CAP).min(to_fetch.len());
                        let mut fetching: Vec<TerrainDataPoint> = vec![default(); end - start];
                        fetching.clone_from_slice(&to_fetch[start..end]);
                        client
                            .send_message::<Channel1, ReqTerrainHeights>(ReqTerrainHeights(
                                fetching,
                            ))
                            .expect("TODO: panic message");
                        println!("Fetched {}", end - start);
                    }
                }
                println!("Fetch {} verts", to_fetch.len());
                terrain_data_map.initialized = true;
                fetch_iter.0 += 1;

                // Task is complete, so remove task component from entity
                commands.entity(entity).remove::<ComputeRequiredData>();
            }
        }
    }
}

pub fn fill_in_fetched_data(
    mut commands: Commands,
    mut terrain_data_map: ResMut<TerrainDataMap>,
    mut fill_in_tasks: Query<&mut FillInData>,
    mut terrain_heights_reader: EventReader<MessageEvent<TerrainHeights>>,
) {
    let mut data = Vec::new();
    for event in terrain_heights_reader.read() {
        data.extend(event.message().0.clone());
    }

    if !data.is_empty() {
        let thread_pool = AsyncComputeTaskPool::get();
        let entity = commands.spawn_empty().id();
        let mut data_map = terrain_data_map.into_inner().clone();
        let task = thread_pool.spawn(async move {
            let res = data_map.fill_in_data(&data);
            return (res, data_map);
        });
        commands.entity(entity).insert(FillInData(task));
    }
}

pub fn update_terrain_vertices(
    mut commands: Commands,
    mut terrain_data_map: ResMut<TerrainDataMap>,
    mut terrain_material: ResMut<Assets<TerrainMaterial>>,
    mut fill_in_tasks: Query<(Entity, &mut FillInData)>,
) {
    let mut updated = false;
    for (entity, mut task) in &mut fill_in_tasks {
        if let Some(res) = block_on(future::poll_once(&mut task.0)) {
            let (did_update, filled) = res;
            // update the terrain heights with this new stuff
            terrain_data_map.unresolved_data = filled.unresolved_data;
            terrain_data_map.resolved_data = filled.resolved_data;
            terrain_data_map.center = filled.center;
            updated = did_update;

            commands.entity(entity).remove::<FillInData>();
        }
    }

    if !updated {
        return;
    }

    let mut delaunay_points: Vec<Point> = Vec::with_capacity(MAX_VERTICES);
    let mut raw: Vec<f32> = Vec::with_capacity(MAX_VERTICES * 2);
    let mut heights: Vec<f32> = Vec::with_capacity(MAX_VERTICES);
    let mut gradients: Vec<f32> = Vec::with_capacity(MAX_VERTICES * 2);

    for datum in &terrain_data_map.resolved_data.data {
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
    }

    let triangulation = triangulate(&delaunay_points);

    // only one material, so it's first
    let (handle, material) = terrain_material.iter_mut().next().unwrap();

    material.vertices = raw;
    material.triangles = triangulation
        .triangles
        .into_iter()
        .map(|e| e as u32)
        .collect();
    material.halfedges = triangulation
        .halfedges
        .into_iter()
        .map(|e| e as u32)
        .collect();
    material.height = heights;
    material.gradients = gradients;
}

pub struct TerrainDataUpdate {
    vertices: Vec<f32>,
    triangules: Vec<u32>,
    halfedges: Vec<u32>,
    heights: Vec<f32>,
    gradients: Vec<f32>,
}

#[derive(Resource, Deref)]
pub struct StreamReceiver(Receiver<TerrainDataUpdate>);

#[derive(Resource)]
pub struct MeshOffsetRes(Arc<Mutex<Transform>>);

pub fn setup_terrain_loader(
    mut commands: Commands,
    transform: Res<MeshOffsetRes>,
    terrain_data_map: Res<TerrainDataMap>,
) {
    let (tx, rx) = bounded::<TerrainDataUpdate>(5);
    let transform = Arc::clone(&transform.0);

    // we just use this once, and it lives ~forever~ (spooky)
    let mut map = terrain_data_map.clone();
    let mesh_verts = terrain_data_map.terrain_mesh_verts.clone();
    tokio::spawn(async move {
        // now fetch from the server!
        let (mut socket, other) = connect_async(
            Url::parse("ws://localhost:9001/getCaseCount")
                .expect("Can't connect to case count URL"),
        )
        .await
        .unwrap();

        loop {
            // update the center
            {
                let t = transform.lock().unwrap();
                map.center = t.translation.xz();
            }

            let batch_size = 512 / size_of::<TerrainDataPoint>();
            let mut to_fetch = Vec::<TerrainDataPoint>::with_capacity(batch_size);

            for i in 0..mesh_verts.len() {
                let vert = &mesh_verts[i];
                // now we check the data map
                if map.should_fetch_more(vert) {
                    to_fetch.push(TerrainDataPoint {
                        coordinates: *vert,
                        height: 0.,
                        idx: 0,
                        gradient: [0., 0.],
                    });
                }

                // send off a load and mark requested
                // we do this so the server can start fetching data for us
                if to_fetch.len() >= batch_size || (i == mesh_verts.len() - 1 && to_fetch.len() > 0)
                {
                    socket
                        .send(Binary(to_stdvec(&to_fetch).unwrap()))
                        .await
                        .expect("TODO: panic message");
                    // now clear and restart
                    map.mark_requested(&mut to_fetch);
                    to_fetch.clear();
                }
            }


            /// send a signal to update our shader once we've received everything we requested
            let mut delaunay_points: Vec<Point> = Vec::with_capacity(MAX_VERTICES);
            let mut raw: Vec<f32> = Vec::with_capacity(MAX_VERTICES * 2);
            let mut heights: Vec<f32> = Vec::with_capacity(MAX_VERTICES);
            let mut gradients: Vec<f32> = Vec::with_capacity(MAX_VERTICES * 2);

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
            }

            let triangulation = triangulate(&delaunay_points);

            tx.send(TerrainDataUpdate {
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
            })
            .unwrap();
        }
    });

    commands.insert_resource(StreamReceiver(rx));
}

pub fn stream_terrain_mesh(
    receiver: Res<StreamReceiver>,
    mut terrain_material: ResMut<Assets<TerrainMaterial>>,
) {
    let last = receiver.try_iter().last();
    if let Some(update) = last {
        let (handle, material) = terrain_material.iter_mut().next().unwrap();
        material.vertices = update.vertices;
        material.triangles = update.triangules;
        material.halfedges = update.halfedges;
        material.height = update.heights;
        material.gradients = update.gradients;
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
    locator: KdTree<f32, 2>,
    /// This is the list of eviction candidates from best to worst
    /// It is sorted by the candidates' ratio of actual distance to ideal distance
    eviction_priority: Vec<(usize, f32)>,
}

#[derive(Resource, Clone)]
pub struct TerrainDataMap {
    initialized: bool,
    resolved_data: TerrainDataHolder,
    unresolved_data: TerrainDataHolder,
    terrain_mesh_verts: Vec<[f32; 2]>,
    /// The reverse locator lets us find which point in the terrain MESH we are closest to quickly
    reverse_locator: ImmutableKdTree<f32, 2>,
    /// Center of the terrain mesh
    center: Vec2,
}

/// How much the nodes are rounded by
const ACCURACY: f32 = 0.25;

impl TerrainDataHolder {
    /// Partially sort the priorities by the first n highest
    pub fn select_highest_priorities(
        &mut self,
        reverse_locator: &ImmutableKdTree<f32, 2>,
        center: &Vec2,
        n_highest: usize,
    ) {
        for (i, data) in (&self.data).into_iter().enumerate() {
            if data.coordinates[0].is_infinite() || data.coordinates[1].is_infinite() {
                self.eviction_priority[i] = (i, f32::INFINITY);
            } else {
                let required_dist = get_check_radius(&Vec2::from(data.coordinates), center);
                let actual_dist =
                    reverse_locator.nearest_one::<SquaredEuclidean>(&data.coordinates);
                self.eviction_priority[i] = (i, actual_dist.distance / required_dist);
            }
        }
        self.eviction_priority
            .select_nth_unstable_by(n_highest, |a, b| b.1.total_cmp(&a.1));
    }
}

impl TerrainDataMap {
    pub fn new(terrain_lattice: &Vec<DVec3>) -> Self {
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
                MAX_VERTICES
            ],
            locator: Default::default(),
            eviction_priority: Vec::from_iter(
                vec![f32::INFINITY; MAX_VERTICES].into_iter().enumerate(),
            ),
        };
        return TerrainDataMap {
            initialized: false,
            resolved_data: holder.clone(),
            unresolved_data: holder,
            reverse_locator: ImmutableKdTree::<f32, 2>::from(&*lattice_data),
            terrain_mesh_verts: lattice_data,
            center: Vec2::ZERO,
        };
    }

    /// Mark these vertices as requested data
    /// We potentially evict other requested data before it's been filled in
    pub fn mark_requested(&mut self, mut data: &mut Vec<TerrainDataPoint>) {
        // add it to the locator to mark it as part of the useful data
        self.unresolved_data.select_highest_priorities(
            &self.reverse_locator,
            &self.center,
            data.len(),
        );
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
                let mut old_data = &mut self.unresolved_data.data[new_data.idx];
                return old_data.coordinates[0] == new_data.coordinates[0]
                    && old_data.coordinates[1] == new_data.coordinates[1];
            })
            .collect_vec();
        // prioritize for replacement
        self.resolved_data.select_highest_priorities(
            &self.reverse_locator,
            &self.center,
            data.len(),
        );
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
            let mut old_resolved = &mut self.resolved_data.data[evicting];
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
    pub fn should_fetch_more(&self, vertex: &[f32; 2]) -> bool {
        let vertex = Vec2::from(*vertex);
        // we check if the nearest guy is in the acceptable range
        let nearest = self
            .resolved_data
            .locator
            .nearest_one::<SquaredEuclidean>(vertex.as_ref());
        let unresolved_nearest = self
            .unresolved_data
            .locator
            .nearest_one::<SquaredEuclidean>(vertex.as_ref());
        return nearest.distance.min(unresolved_nearest.distance)
            > get_check_radius(&vertex, &self.center);
    }
}
