use bevy::ecs::system::CommandQueue;
// use std::net::TcpStream;
use crate::terrain_render::*;
use bevy::math::DVec2;
use bevy::prelude::*;
use bevy::render::mesh::VertexAttributeValues::Float32x3;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use futures_util::SinkExt;
use kiddo::KdTree;
use server::terrain_data::TerrainTile;
// use tokio_tungstenite::MaybeTlsStream;
use crate::player_controller::{PlayerBody, PlayerCam};
use tokio_tungstenite_wasm::*;

/// Quarter meter data tolerance when the player is 1m away
const DATA_TOLERANCE: f32 = 0.25;

#[derive(Resource)]
struct ServerConnection(WebSocketStream);

#[derive(Resource)]
struct TerrainData(KdTree<f64, 2>);

#[derive(Component)]
struct FetchHeightDataTask(Task<CommandQueue>);

fn request_terrain_vertices(
    mut commands: Commands,
    terrain_transform: Query<
        &mut Transform,
        (With<MainTerrain>, Without<PlayerBody>, Without<PlayerCam>),
    >,
    terrain_mesh: Query<&Handle<Mesh>, With<MainTerrain>>,
    meshes: Res<Assets<Mesh>>,
    terrain_data_map: ResMut<TerrainDataMap>,
    terrain_data: ResMut<TerrainData>,
    mut socket: ResMut<ServerConnection>,
) {
    let thread_pool = AsyncComputeTaskPool::get();
    // we get which vertices need to be fetched
    let terrain_handle = terrain_mesh.get_single().unwrap();
    let terrain = meshes.get(terrain_handle).unwrap();
    let terrain_tf = terrain_transform.get_single().unwrap();
    if let Some(Float32x3(verts)) = terrain.attribute(Mesh::ATTRIBUTE_POSITION) {
        for vert in verts {
            let orig = Vec3::from(*vert);
            let actual_pos = orig + terrain_tf.translation;
            let origin_dist = orig.length();
            let check_radius = origin_dist * DATA_TOLERANCE;
            // now we check the data map
            if !terrain_data_map.data_exists(Square::new(
                &DVec2::from(actual_pos.xz()).to_array(),
                check_radius as f64,
            )) {
                // we need to fetch the data!
                // make a call to the server?
                let entity = commands.spawn_empty().id();
                let task = thread_pool.spawn(async move {
                    socket.0.send(Message::Binary())
                });
                // Spawn new entity and add our new task as a component
                commands.entity(entity).insert(FetchHeightDataTask(task));
            }
        }
    }
}

// pub async fn get_height(
//     chunk_coords: (i32, i32),
//     center_coords: (i32, i32),
//     socket: &mut WebSocketStream<MaybeTlsStream<TcpStream>>,
// ) -> TerrainTile {
//     let lod = get_lod(chunk_coords, center_coords);
//     socket
//         .send(Message::Text(format!(
//             "{},{}:{lod}",
//             chunk_coords.0, chunk_coords.1
//         )))
//         .await
//         .expect("Failed to send for chunk");
//     let msg = socket.next().await.expect("Can't fetch chunk");
//     return TerrainTile::from(msg.unwrap().into_data());
// }

pub fn get_required_data(socket: &mut WebSocketStream) -> Vec<(i32, i32)> {
    let mut to_fetch = vec![];
    /*
    // iterate over all the vertices in our terrain mesh
    // find the closest point
    // if it's not close enough, fetch a new one
    for lattice_pos in lattice {
        let world_pos = (center.0 + lattice_pos.0, center.1 + lattice_pos.1);
        let nearest = data.nearest_one::<SquaredEuclidean>(&[world_pos.0, world_pos.1]);
        // TODO: make non-static distance
        if nearest.distance > 1.0 {
            to_fetch.push(world_pos);
        }
    }*/
    return to_fetch;
}

#[derive(Resource)]
pub struct TerrainDataMap {

}

pub struct TerrainDataPoint {
    coordinates: [f64; 2],
    height: f64,
}

pub struct Square {
    side: i32,
    top_left: [i32; 2],
}

/// How much the nodes are rounded by
const ACCURACY: f64 = 0.25;

impl TerrainDataMap {
    pub fn new() -> Self {
        return TerrainDataMap {
            data_existence: HashSet::new(),
            height_data: HashMap::new(),
        };
    }

    pub fn insert(&mut self, data: TerrainDataPoint) {
        // unpack the coordinates
        let [x, y] = data.coordinates;
        // convert to pos on smallest grid
        let [x, y] = [(x / ACCURACY).round() as i32, (y / ACCURACY).round() as i32];
        /// we insert ourselves at every level
        /// technically not necessary, but easy
        for level in 0..=TREE_DEPTH {
            let cell_size = TREE_LARGE_NODE >> level;
            // convert to pos on current grid
            let [x, y] = [x.div_euclid(cell_size), y.div_euclid(cell_size)];
            self.data_existence.insert(([x, y], level));
        }
        self.height_data.insert([x, y], data);
    }

    pub fn remove(&mut self, coords: &[f64; 2]) {
        // unpack the coordinates
        let [x, y] = coords;
        // convert to pos on smallest grid
        let [x, y] = [(x / ACCURACY).round() as i32, (y / ACCURACY).round() as i32];
        // remove data point
        self.height_data.remove(&[x, y]);
        self.data_existence.remove(&([x, y], TREE_DEPTH));
        for level in (0..TREE_DEPTH).rev() {
            // check the children at each level
            let cell_size = TREE_LARGE_NODE >> level;
            // convert to pos on current grid
            let [x, y] = [x.div_euclid(cell_size), y.div_euclid(cell_size)];
            let mut exists = false;
            for i in 0..2 {
                for j in 0..2 {
                    let cell = ([x * 2 + i, y * 2 + j], level + 1);
                    exists |= self.data_existence.contains(&cell);
                }
            }
            if !exists {
                self.data_existence.remove(&([x, y], level));
            }
        }
    }

    /// Determine if terrain data exists within the square
    /// We don't need to to know what the data is, only if it exists
    pub fn data_exists(&self, square: Square) -> bool {
        // get all the top level squares that we can
        let mut to_check = VecDeque::<([i32; 2], i32)>::new();
        let horiz_beg = square.top_left[0].div_euclid(TREE_LARGE_NODE);
        let horiz_end = (square.top_left[0] + square.side).div_euclid(TREE_LARGE_NODE);
        let vert_beg = square.top_left[1].div_euclid(TREE_LARGE_NODE);
        let vert_end = (square.top_left[1] + square.side).div_euclid(TREE_LARGE_NODE);
        for x in horiz_beg..=horiz_end {
            for y in vert_beg..=vert_end {
                let cell = ([x, y], 0);
                if self.data_existence.contains(&cell) {
                    if square.contains(&cell) {
                        return true;
                    }
                    // it's not entirely contained, so check it's children
                    to_check.push_back(([x, y], 0));
                }
            }
        }
        while !to_check.is_empty() {
            let ([x, y], depth) = to_check.pop_front().unwrap();
            if depth >= TREE_DEPTH {
                continue;
            }
            // check the four children of this cell
            for i in 0..2 {
                for j in 0..2 {
                    let cell = ([x * 2 + i, y * 2 + j], depth + 1);
                    if self.data_existence.contains(&cell) {
                        if square.contains(&cell) {
                            // if it's entirely contained and contains data, we can return early
                            return true;
                        }
                        // it's not entirely contained, so check it's children
                        to_check.push_back(cell);
                    }
                }
            }
        }
        return false;
    }
}

#[test]
fn terrain_data_map_insert_test() {
    let mut dmap = TerrainDataMap::new();
    dmap.insert(TerrainDataPoint {
        coordinates: [0., 0.],
        height: 1000.,
    });

    dmap.insert(TerrainDataPoint {
        coordinates: [0.33, 0.33],
        height: 1000.,
    });
    for ([x, y], d) in dmap.data_existence {
        println!("{} [{}, {}]", d, x, y);
    }
}

#[test]
fn terrain_data_remove_test() {
    let mut dmap = TerrainDataMap::new();
    dmap.insert(TerrainDataPoint {
        coordinates: [0., 0.],
        height: 1000.,
    });

    dmap.insert(TerrainDataPoint {
        coordinates: [0.33, 0.33],
        height: 1000.,
    });

    {
        assert_eq!(dmap.data_existence.len(), 22);
    }
    dmap.remove(&[0.33, 0.33]);
    {
        assert_eq!(dmap.data_existence.len(), 21)
    }
    dmap.remove(&[0., 0.]);
    {
        assert_eq!(dmap.data_existence.len(), 0)
    }
    for ([x, y], d) in dmap.data_existence {
        println!("{} [{}, {}]", d, x, y);
    }
}

#[test]
fn terrain_data_check_test() {
    let mut dmap = TerrainDataMap::new();
    dmap.insert(TerrainDataPoint {
        coordinates: [0., 0.],
        height: 1000.,
    });

    dmap.insert(TerrainDataPoint {
        coordinates: [0.33, 0.33],
        height: 1000.,
    });
    assert!(dmap.data_exists(Square {
        side: 10,
        top_left: [0, 0],
    }));

    assert!(!dmap.data_exists(Square {
        side: 14,
        top_left: [-15, -15],
    }));

    dmap.insert(TerrainDataPoint {
        coordinates: [-3.26, -3.268],
        height: 1000.,
    });
    for ([x, y], d) in &dmap.data_existence {
        let sqr = Square {
            side: 14,
            top_left: [-15, -15],
        };
        println!("{} [{}, {}] {}", d, x, y, sqr.contains(&([*x, *y], *d)));
    }

    assert!(dmap.data_exists(Square {
        side: 14,
        top_left: [-15, -15],
    }));
}